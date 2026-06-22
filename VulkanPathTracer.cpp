// VulkanPathTracer.cpp — full Vulkan + Win32 implementation of the renderer.
//
// File layout (search the SECTION banners below):
//   * helpers          — small utilities (error wrappers, math types,
//                        buffer/descriptor structs).
//   * window procedure — Win32 input plumbing for the camera controls.
//   * Vulkan app       — the VulkanApp class: device setup, swapchain,
//                        scene/Mie buffer upload, compute pipeline, the
//                        render loop, and teardown.
//
// This is a polarized-sky simulator with no scene geometry: a single compute
// shader evaluates the sky analytically per pixel. High-level flow when
// RunVulkanPathTracer() is called:
//   1. Read path_tracer_config.json.
//   2. Create a Win32 window, Vulkan instance, surface, device, queues.
//   3. Upload the sky uniform buffer and bake the Lorenz–Mie SSBO.
//   4. Create the swapchain (storage-image capable), descriptor sets, and the
//      compute pipeline from path_tracer.comp.
//   5. Drive the message loop: each frame integrates camera input and
//      dispatches the compute shader, which writes the swapchain image
//      directly and is then presented.

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#define VK_USE_PLATFORM_WIN32_KHR

#include "VulkanPathTracer.h"

#include "MieScattering.h"
#include "RuntimeConfig.h"

#include <windows.h>
#include <vulkan/vulkan_raii.hpp>

#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

// =====================================================================
// SECTION: helpers
//
// Cross-cutting utilities used by the rest of the file: error reporting
// wrappers around the Win32 BOOL / Vulkan VkResult conventions, SPIR-V
// blob loading, alignment math, and the small POD structs that describe
// queue families, swapchain support, GPU buffers, and acceleration
// structures. Also defines the std140-friendly upload structs (SceneData,
// InstanceData, MaterialData) shared with the shaders — the static_assert
// lines lock the layouts so any accidental field shuffle fails the build.
// =====================================================================

// Throw a std::runtime_error on a failed Win32 BOOL return so call sites
// can treat any non-zero result as success without nested if-blocks.
static void ThrowIfFalse(BOOL condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

// Map any non-success VkResult to a std::runtime_error tagged with the
// numeric result code, so the user sees both the operation that failed
// and the Vulkan-level reason.
static void ThrowVk(VkResult result, const char* message)
{
    if (result != VK_SUCCESS)
    {
        throw std::runtime_error(std::format("{} (VkResult={})", message, static_cast<int>(result)));
    }
}

// Load a SPIR-V module (or any binary blob) from disk into a std::vector.
// Resolution mirrors the runtime config: search exe dir → exe parent →
// CWD; first hit wins. The file is opened in binary mode so the size from
// tellg() matches the actual byte count.
static std::vector<char> LoadBinaryFile(const wchar_t* fileName)
{
    const auto shaderPath = ResolveRuntimeFilePath(fileName);
    if (shaderPath.empty())
    {
        throw std::runtime_error("Failed to open SPIR-V shader.");
    }

    std::ifstream file(shaderPath, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("Failed to open SPIR-V shader.");
    }
    const auto fileSize = file.tellg();
    if (fileSize <= 0)
    {
        throw std::runtime_error("SPIR-V shader is empty.");
    }

    std::vector<char> data(static_cast<size_t>(fileSize));
    file.seekg(0, std::ios::beg);
    file.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (!file)
    {
        throw std::runtime_error("Failed to read SPIR-V shader.");
    }

    return data;
}

// Indices of the queue families this app needs. Optional because they're
// discovered one at a time during physical-device selection.
struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool IsComplete() const
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// Generic GPU buffer + its VMA allocation. The allocation is created
// persistently mapped, so `mapped` points at host-visible memory for the
// buffer's lifetime and uploads are a plain memcpy + flush.
struct BufferAllocation
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

// Sky / scene uniforms uploaded to the compute pipeline.
//
// Each vec4 packs multiple scalars together to keep the std140 layout
// compact and to match the shader-side struct one-for-one. The static
// assertion below pins the size so any future field reshuffle that breaks
// alignment fails to compile rather than corrupting the GPU view of the data.
struct alignas(16) SceneData
{
    float skyBetaRayleighBetaM[4];
    float skyMieEarthAtmosScaleHr[4];
    float skyScaleHmSunRadiusAa[4];
    float skySunRadiance[4];
    float skySunDirection[4];
    uint32_t skySampleCounts[4];
    // Vector radiative transfer: x = Rayleigh depolarization, y unused,
    // z = Mie table angle bins, w = ozone layer Gaussian width (m). The Mie
    // scattering matrix itself rides in the binding-7 SSBO, not here.
    float skyVrtParams[4];
    // Ozone Chappuis-band absorption: xyz = peak absorption coefficient per
    // RGB band (1/m), w = layer center altitude (m).
    float skyOzoneBeta[4];
    // xyz = sun limb-darkening coefficient per RGB band, w = atmospheric
    // refraction strength (1 = standard atmosphere, 0 = off).
    float skySunLimbRefraction[4];
    // Aerosol extras: x = stratospheric background peak extinction (1/m),
    // y = background layer center altitude (m), z = layer Gaussian width (m),
    // w = aerosol single-scattering albedo.
    float skyMieBackground[4];
};

static_assert(sizeof(SceneData) == 160, "Scene data layout must stay 16-byte aligned.");

// 3x3 row-major matrix used only for camera basis math.
struct Mat3
{
    float m[3][3]{};
};

static Mat3 IdentityMat3()
{
    Mat3 result{};
    result.m[0][0] = 1.0f;
    result.m[1][1] = 1.0f;
    result.m[2][2] = 1.0f;
    return result;
}

static Mat3 Multiply(const Mat3& left, const Mat3& right)
{
    Mat3 result{};
    for (uint32_t row = 0; row < 3; ++row)
    {
        for (uint32_t column = 0; column < 3; ++column)
        {
            result.m[row][column] = left.m[row][0] * right.m[0][column]
                                    + left.m[row][1] * right.m[1][column]
                                    + left.m[row][2] * right.m[2][column];
        }
    }
    return result;
}

static Mat3 Transpose(const Mat3& matrix)
{
    Mat3 result{};
    for (uint32_t row = 0; row < 3; ++row)
    {
        for (uint32_t column = 0; column < 3; ++column)
        {
            result.m[row][column] = matrix.m[column][row];
        }
    }
    return result;
}

static Vec3 TransformDirection(const Mat3& matrix, const Vec3& value)
{
    return {
        matrix.m[0][0] * value.x + matrix.m[0][1] * value.y + matrix.m[0][2] * value.z,
        matrix.m[1][0] * value.x + matrix.m[1][1] * value.y + matrix.m[1][2] * value.z,
        matrix.m[2][0] * value.x + matrix.m[2][1] * value.y + matrix.m[2][2] * value.z,
    };
}

static Mat3 MakeScaleMatrix(const Vec3& scale)
{
    Mat3 result{};
    result.m[0][0] = scale.x;
    result.m[1][1] = scale.y;
    result.m[2][2] = scale.z;
    return result;
}

static Mat3 MakeRotationX(float radians)
{
    Mat3 result = IdentityMat3();
    const float cosine = std::cos(radians);
    const float sine = std::sin(radians);
    result.m[1][1] = cosine;
    result.m[1][2] = -sine;
    result.m[2][1] = sine;
    result.m[2][2] = cosine;
    return result;
}

static Mat3 MakeRotationY(float radians)
{
    Mat3 result = IdentityMat3();
    const float cosine = std::cos(radians);
    const float sine = std::sin(radians);
    result.m[0][0] = cosine;
    result.m[0][2] = sine;
    result.m[2][0] = -sine;
    result.m[2][2] = cosine;
    return result;
}

static Mat3 MakeRotationZ(float radians)
{
    Mat3 result = IdentityMat3();
    const float cosine = std::cos(radians);
    const float sine = std::sin(radians);
    result.m[0][0] = cosine;
    result.m[0][1] = -sine;
    result.m[1][0] = sine;
    result.m[1][1] = cosine;
    return result;
}

static Mat3 MakeEulerRotationMatrixDegrees(const Vec3& rotationDegrees)
{
    const float radiansX = rotationDegrees.x * kPi / 180.0f;
    const float radiansY = rotationDegrees.y * kPi / 180.0f;
    const float radiansZ = rotationDegrees.z * kPi / 180.0f;
    return Multiply(Multiply(MakeRotationZ(radiansZ), MakeRotationY(radiansY)), MakeRotationX(radiansX));
}

static bool HasSkySpectralChanged(const SkySpectralConfig& left, const SkySpectralConfig& right)
{
    return left.betaRayleigh != right.betaRayleigh
           || left.betaMie != right.betaMie
           || left.mieG != right.mieG
           || left.earthRadius != right.earthRadius
           || left.atmosphereRadius != right.atmosphereRadius
           || left.scaleHeightRayleigh != right.scaleHeightRayleigh
           || left.scaleHeightMie != right.scaleHeightMie
           || left.sunRadiance != right.sunRadiance
           || left.sunRadius != right.sunRadius
           || left.sunAa != right.sunAa
           || left.betaOzone != right.betaOzone
           || left.ozoneCenterAltitude != right.ozoneCenterAltitude
           || left.ozoneLayerWidth != right.ozoneLayerWidth
           || left.sunLimbDarkening != right.sunLimbDarkening
           || left.refractionStrength != right.refractionStrength
           || left.mieBackgroundBeta != right.mieBackgroundBeta
           || left.mieBackgroundCenter != right.mieBackgroundCenter
           || left.mieBackgroundWidth != right.mieBackgroundWidth
           || left.mieSingleScatterAlbedo != right.mieSingleScatterAlbedo
           || left.secondarySamples != right.secondarySamples
           || left.viewSteps != right.viewSteps
           || left.samples != right.samples
           || left.rayleighDepolarization != right.rayleighDepolarization
           || HasMieAerosolChanged(left, right);
}

struct PushConstants
{
    float cameraPositionFrame[4];
    float cameraForwardSamples[4];
    float cameraRightBounces[4];
    float cameraUpTanHalfFovY[4];
    float skyBottomExposure[4];
    float skyTopAspect[4];
    // x = filter enabled (0/1), y = filter axis angle in radians, z/w unused.
    float polarizer[4];
    uint32_t imageSize[2];
};

static_assert(sizeof(PushConstants) == 120, "Push constant layout must match the shader.");

// =====================================================================
// SECTION: window procedure
//
// Win32 message handling for the render window. The path tracer takes raw
// keyboard / mouse input directly (no input library) so the WndProc here
// just translates messages into flags on a small InputState struct that
// the render loop polls each frame. Mouse-look uses raw input so motion
// is independent of the desktop pointer-acceleration curve.
// =====================================================================

static LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(window, message, wParam, lParam);
    }
}

// =====================================================================
// SECTION: Vulkan app
//
// VulkanApp owns every Vulkan handle the renderer needs and drives the
// frame loop. The constructor walks through Vulkan setup top-to-bottom:
// instance + surface + device, command pools, the sky uniform buffer and
// baked Lorenz–Mie SSBO, swapchain, descriptor sets, and finally the
// compute pipeline.
//
// Render() runs once per frame: poll the camera, push fresh uniforms, and
// dispatch the compute shader, which writes the acquired swapchain image
// directly; the image is then presented.
// All resources are released in the destructor in reverse construction
// order so partial-init failure paths (which throw mid-construction)
// still tear down cleanly via RAII member destructors where possible.
// =====================================================================

// Owns every Vulkan handle and orchestrates the renderer. Public surface
// is intentionally thin (just Run()); construction is two-phase via Run()
// rather than the ctor so failures can throw with cleanup driven by
// Cleanup() rather than partial-RAII destructors.
class VulkanPathTracer
{
public:
    // Default scene/render configuration file, resolved relative to the
    // executable by ResolveRuntimeFilePath().
    static constexpr const wchar_t* CONFIG_FILE_NAME = L"path_tracer_config.json";

    // RAII member destructors release every Vulkan handle in reverse
    // declaration order. The only manual teardown is the VMA allocator and its
    // buffers, which must be released here (in the destructor body, before any
    // raii member is destroyed) while the device is still alive.
    ~VulkanPathTracer()
    {
        if (*m_device)
        {
            m_device.waitIdle();
        }
        DestroySceneBuffers();
        if (m_allocator != VK_NULL_HANDLE)
        {
            vmaDestroyAllocator(m_allocator);
            m_allocator = VK_NULL_HANDLE;
        }
        if (m_window != nullptr)
        {
            DestroyWindow(m_window);
        }
    }

    // Top-level driver: load config → init Vulkan → render until the
    // window closes → tear everything down. Throws on any failure; the
    // caller in main() converts that into a non-zero exit code.
    void Run()
    {
        LoadInitialRuntimeConfig();
        CreateWindowAndShow();
        CreateInstance();
        CreateSurface();
        PickPhysicalDevice();
        CreateLogicalDevice();
        CreateAllocator();
        CreateCommandPool();
        CreateSceneResources();
        CreateSwapchain();
        CreateDescriptorSetLayout();
        CreatePipeline();
        CreateDescriptorSets();
        CreateCommandBuffers();
        CreateSyncObjects();
        MessageLoop();
        m_device.waitIdle();
    }

private:
    static bool IsKeyDown(int virtualKey)
    {
        return (GetAsyncKeyState(virtualKey) & 0x8000) != 0;
    }

    float GetCameraMaxPitchRadians() const
    {
        return m_config.maxPitchDegrees * kPi / 180.0f;
    }

    SceneData BuildSceneData() const
    {
        const SkySpectralConfig& s = m_config.skySpectral;
        SceneData sceneData{};
        sceneData.skyBetaRayleighBetaM[0] = s.betaRayleigh[0];
        sceneData.skyBetaRayleighBetaM[1] = s.betaRayleigh[1];
        sceneData.skyBetaRayleighBetaM[2] = s.betaRayleigh[2];
        sceneData.skyBetaRayleighBetaM[3] = s.betaMie;
        sceneData.skyMieEarthAtmosScaleHr[0] = s.mieG;
        sceneData.skyMieEarthAtmosScaleHr[1] = s.earthRadius;
        sceneData.skyMieEarthAtmosScaleHr[2] = s.atmosphereRadius;
        sceneData.skyMieEarthAtmosScaleHr[3] = s.scaleHeightRayleigh;
        sceneData.skyScaleHmSunRadiusAa[0] = s.scaleHeightMie;
        sceneData.skyScaleHmSunRadiusAa[1] = s.sunRadius;
        sceneData.skyScaleHmSunRadiusAa[2] = s.sunAa;
        sceneData.skySunRadiance[0] = s.sunRadiance[0];
        sceneData.skySunRadiance[1] = s.sunRadiance[1];
        sceneData.skySunRadiance[2] = s.sunRadiance[2];
        sceneData.skySunDirection[0] = s.sunDirection[0];
        sceneData.skySunDirection[1] = s.sunDirection[1];
        sceneData.skySunDirection[2] = s.sunDirection[2];
        sceneData.skySampleCounts[0] = s.secondarySamples;
        sceneData.skySampleCounts[1] = s.viewSteps;
        sceneData.skySampleCounts[2] = s.samples;
        sceneData.skyVrtParams[0] = s.rayleighDepolarization;
        sceneData.skyVrtParams[1] = 0.0f;
        sceneData.skyVrtParams[2] = static_cast<float>(s.mieTableAngleBins);
        sceneData.skyVrtParams[3] = s.ozoneLayerWidth;
        sceneData.skyOzoneBeta[0] = s.betaOzone[0];
        sceneData.skyOzoneBeta[1] = s.betaOzone[1];
        sceneData.skyOzoneBeta[2] = s.betaOzone[2];
        sceneData.skyOzoneBeta[3] = s.ozoneCenterAltitude;
        sceneData.skySunLimbRefraction[0] = s.sunLimbDarkening[0];
        sceneData.skySunLimbRefraction[1] = s.sunLimbDarkening[1];
        sceneData.skySunLimbRefraction[2] = s.sunLimbDarkening[2];
        sceneData.skySunLimbRefraction[3] = s.refractionStrength;
        sceneData.skyMieBackground[0] = s.mieBackgroundBeta;
        sceneData.skyMieBackground[1] = s.mieBackgroundCenter;
        sceneData.skyMieBackground[2] = s.mieBackgroundWidth;
        sceneData.skyMieBackground[3] = s.mieSingleScatterAlbedo;
        return sceneData;
    }

    // Allocate the scene uniform buffer (sky parameters) and the Lorenz–Mie
    // scattering-matrix SSBO consumed by the ray-generation shader.
    void CreateSceneBuffers()
    {
        m_sceneDataBuffer = CreateBuffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        CreateMieScatteringBuffer();
    }

    // Translate the aerosol fields of the live config into the Mie precompute
    // parameters.
    MieAerosolParams BuildMieAerosolParams() const
    {
        const SkySpectralConfig& s = m_config.skySpectral;
        MieAerosolParams params{};
        params.refractiveIndexReal = s.aerosolRefractiveIndexReal;
        params.refractiveIndexImag = s.aerosolRefractiveIndexImag;
        params.meanRadiusMicrometers = s.aerosolMeanRadiusMicrometers;
        params.sigma = s.aerosolSigma;
        params.wavelengthsNmRgb[0] = s.aerosolWavelengthsNmRgb[0];
        params.wavelengthsNmRgb[1] = s.aerosolWavelengthsNmRgb[1];
        params.wavelengthsNmRgb[2] = s.aerosolWavelengthsNmRgb[2];
        params.angleBins = static_cast<int>(s.mieTableAngleBins);
        return params;
    }

    // Bake the Lorenz–Mie scattering matrix on the CPU and stage it into the
    // binding-7 SSBO sampled by the polarized sky integrator. The table is
    // immutable for the buffer's lifetime, so it is uploaded here rather than
    // through UploadSceneDataFromConfig.
    void CreateMieScatteringBuffer()
    {
        const MieAerosolParams params = BuildMieAerosolParams();
        const std::vector<MieMatrixEntry> table = ComputeMieScatteringTable(params);
        const VkDeviceSize size = static_cast<VkDeviceSize>(table.size() * sizeof(MieMatrixEntry));
        m_mieScatteringBuffer = CreateBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        UploadToBuffer(m_mieScatteringBuffer, std::as_bytes(std::span{table}));
        std::printf("[Sky] Baked Lorenz-Mie scattering matrix: %d angle bins x 3 bands.\n", params.angleBins);
    }

    // Release the uniform buffer created by CreateSceneBuffers(). Safe to
    // call when the buffer was never created (no-op).
    void DestroySceneBuffers()
    {
        DestroyBuffer(m_sceneDataBuffer);
        DestroyBuffer(m_mieScatteringBuffer);
    }

    // Pack the current RuntimeConfig.skySpectral into a SceneData record and
    // stage it into the scene UBO. Called whenever the config is (re)loaded.
    void UploadSceneDataFromConfig()
    {
        const SceneData sceneData = BuildSceneData();
        UploadToBuffer(m_sceneDataBuffer, std::as_bytes(std::span{&sceneData, 1}));
    }

    // Bind the live resources (output image, scene UBO, Mie SSBO) into the
    // descriptor sets.
    void UpdateDescriptorSetContents()
    {
        if (m_descriptorSets.empty())
        {
            return;
        }

        for (size_t i = 0; i < m_descriptorSets.size(); ++i)
        {
            const vk::DescriptorImageInfo imageInfo{{}, *m_swapchainImageViews[i], vk::ImageLayout::eGeneral};
            const vk::DescriptorBufferInfo sceneDataInfo{m_sceneDataBuffer.buffer, 0, m_sceneDataBuffer.size};
            const vk::DescriptorBufferInfo mieDataInfo{m_mieScatteringBuffer.buffer, 0, m_mieScatteringBuffer.size};

            vk::WriteDescriptorSet imageWrite{*m_descriptorSets[i], 0, 0, vk::DescriptorType::eStorageImage, imageInfo};
            vk::WriteDescriptorSet sceneWrite{*m_descriptorSets[i], 2, 0, vk::DescriptorType::eUniformBuffer, {}, sceneDataInfo};
            vk::WriteDescriptorSet mieWrite{*m_descriptorSets[i], 7, 0, vk::DescriptorType::eStorageBuffer, {}, mieDataInfo};

            m_device.updateDescriptorSets({imageWrite, sceneWrite, mieWrite}, nullptr);
        }
    }

    // Re-apply the current m_config's sky parameters to live GPU resources.
    // Rebuilds the Mie scattering table when the aerosol model changed, then
    // re-uploads the scene UBO.
    void RefreshSceneFromConfig(bool rebuildMieTable)
    {
        if (m_sceneDataBuffer.buffer == VK_NULL_HANDLE)
        {
            return;
        }

        m_device.waitIdle();
        if (rebuildMieTable)
        {
            DestroyBuffer(m_mieScatteringBuffer);
            CreateMieScatteringBuffer();
            UpdateDescriptorSetContents();
        }
        UploadSceneDataFromConfig();
    }

    // Diff the incoming config against the current one and apply the cheapest
    // valid refresh (camera reset, Mie rebuild, sky UBO re-upload).
    void ApplyRuntimeConfig(const RuntimeConfig& config, bool resetCameraState)
    {
        const bool skySpectralChanged = HasSkySpectralChanged(config.skySpectral, m_config.skySpectral);
        const bool mieAerosolChanged = HasMieAerosolChanged(config.skySpectral, m_config.skySpectral);

        if (!m_configPath.empty() && !resetCameraState)
        {
            if (config.width != m_config.width || config.height != m_config.height
                || config.frameCount != m_config.frameCount)
            {
                std::puts("[Config] width/height/frameCount changes apply on the next launch.");
            }
        }

        m_config = config;
        if (resetCameraState)
        {
            ResetCamera();
            return;
        }

        const float maxPitch = GetCameraMaxPitchRadians();
        m_cameraPitch = std::clamp(m_cameraPitch, -maxPitch, maxPitch);

        if (skySpectralChanged)
        {
            RefreshSceneFromConfig(mieAerosolChanged);
        }
    }

    // Load and parse path_tracer_config.json from disk for the first
    // time. Records the file's last-write time so ReloadRuntimeConfigIfNeeded
    // can pick up live edits.
    void LoadInitialRuntimeConfig()
    {
        m_configPath = ResolveRuntimeFilePath(CONFIG_FILE_NAME);
        if (m_configPath.empty())
        {
            throw std::runtime_error("Failed to locate path_tracer_config.json.");
        }

        ApplyRuntimeConfig(ParseRuntimeConfig(LoadTextFile(m_configPath)), true);

        std::error_code errorCode;
        m_configLastWriteTime = std::filesystem::last_write_time(m_configPath, errorCode);
        if (errorCode)
        {
            throw std::runtime_error("Failed to read config file timestamp.");
        }

        std::printf("[Config] Loaded %s\n", m_configPath.string().c_str());
    }

    // Poll the config file's last-write time once per frame; on a change
    // re-parse it and ApplyRuntimeConfig() the result. Failures during
    // reload are logged and ignored so a typo in the JSON does not crash
    // an interactive editing session.
    void ReloadRuntimeConfigIfNeeded()
    {
        if (m_configPath.empty())
        {
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now - m_lastConfigPollTime < std::chrono::milliseconds(250))
        {
            return;
        }
        m_lastConfigPollTime = now;

        std::error_code errorCode;
        const auto currentWriteTime = std::filesystem::last_write_time(m_configPath, errorCode);
        if (errorCode || currentWriteTime == m_configLastWriteTime)
        {
            return;
        }

        try
        {
            ApplyRuntimeConfig(ParseRuntimeConfig(LoadTextFile(m_configPath)), false);
            std::printf("[Config] Reloaded %s\n", m_configPath.string().c_str());
        }
        catch (const std::exception& error)
        {
            std::fprintf(stderr, "[Config] Reload failed: %s\n", error.what());
        }

        m_configLastWriteTime = currentWriteTime;
    }

    // Snap the camera back to the configured initialPosition / initialLookAt
    // and recompute its yaw/pitch.
    void ResetCamera()
    {
        m_cameraPosition = m_config.initialPosition;
        const Vec3 initialForward = Normalize(m_config.initialLookAt - m_cameraPosition);
        m_cameraYaw = std::atan2(initialForward.x, initialForward.z);
        m_cameraPitch = std::asin(std::clamp(initialForward.y, -1.0f, 1.0f));
    }

    Vec3 GetCameraForward() const
    {
        const float cosPitch = std::cos(m_cameraPitch);
        return Normalize({
            std::sin(m_cameraYaw) * cosPitch,
            std::sin(m_cameraPitch),
            std::cos(m_cameraYaw) * cosPitch,
        });
    }

    // Consume accumulated mouse-delta input and rotate the camera. Pitch
    // is clamped to ±maxPitchDegrees so the camera never goes upside
    // down and the yaw axis remains world-up.
    void UpdateMouseLook()
    {
        const bool windowFocused = GetForegroundWindow() == m_window;
        const bool wantsMouseLook = windowFocused && IsKeyDown(VK_RBUTTON);
        if (!wantsMouseLook)
        {
            if (m_mouseLookActive && GetCapture() == m_window)
            {
                ReleaseCapture();
            }
            m_mouseLookActive = false;
            return;
        }

        POINT cursorPosition{};
        if (!GetCursorPos(&cursorPosition))
        {
            return;
        }

        if (!m_mouseLookActive)
        {
            m_mouseLookActive = true;
            m_lastMousePosition = cursorPosition;
            SetCapture(m_window);
            return;
        }

        const float mouseDeltaX = static_cast<float>(cursorPosition.x - m_lastMousePosition.x);
        const float mouseDeltaY = static_cast<float>(cursorPosition.y - m_lastMousePosition.y);
        m_lastMousePosition = cursorPosition;

        const float maxPitch = GetCameraMaxPitchRadians();
        m_cameraYaw += mouseDeltaX * m_config.mouseSensitivity;
        m_cameraPitch = std::clamp(m_cameraPitch - mouseDeltaY * m_config.mouseSensitivity, -maxPitch, maxPitch);
    }

    // Integrate look input over deltaSeconds: mouse-look, arrow-key look, and
    // the R reset. The sky is directional, so the camera only rotates — there
    // is no positional movement.
    void UpdateCamera(double deltaSeconds)
    {
        const float deltaTime = static_cast<float>(std::min(deltaSeconds, 0.1));
        UpdateMouseLook();

        const bool windowFocused = GetForegroundWindow() == m_window;
        const bool resetCameraDown = windowFocused && IsKeyDown('R');
        if (resetCameraDown && !m_resetCameraKeyDown)
        {
            ResetCamera();
        }
        m_resetCameraKeyDown = resetCameraDown;

        // Polarization filter: P toggles it (edge-triggered so one press is
        // one toggle), [ and ] rotate the filter axis while held.
        const bool polarizerToggleDown = windowFocused && IsKeyDown('P');
        if (polarizerToggleDown && !m_polarizerToggleKeyDown)
        {
            m_polarizerEnabled = !m_polarizerEnabled;
            std::printf("[Polarizer] %s (%s)\n",
                        m_polarizerEnabled ? "ON" : "OFF",
                        m_polarizerElliptical ? "elliptical" : "linear");
        }
        m_polarizerToggleKeyDown = polarizerToggleDown;

        // C toggles between a linear analyzer and an elliptical one.
        const bool polarizerModeDown = windowFocused && IsKeyDown('C');
        if (polarizerModeDown && !m_polarizerModeKeyDown)
        {
            m_polarizerElliptical = !m_polarizerElliptical;
            std::printf("[Polarizer] mode: %s\n", m_polarizerElliptical ? "elliptical" : "linear");
        }
        m_polarizerModeKeyDown = polarizerModeDown;

        if (windowFocused)
        {
            if (m_polarizerElliptical)
            {
                // [ / ] adjust ellipticity; +/-45 degrees reaches circular.
                if (IsKeyDown(VK_OEM_4))
                {
                    m_polarizerEllipticityRadians -= m_config.polarizerRotateSpeed * deltaTime;
                }
                if (IsKeyDown(VK_OEM_6))
                {
                    m_polarizerEllipticityRadians += m_config.polarizerRotateSpeed * deltaTime;
                }
                m_polarizerEllipticityRadians = std::clamp(m_polarizerEllipticityRadians, -kPi * 0.25f, kPi * 0.25f);
            }
            else
            {
                if (IsKeyDown(VK_OEM_4)) // '[' rotates the filter axis one way
                {
                    m_polarizerAngleRadians -= m_config.polarizerRotateSpeed * deltaTime;
                }
                if (IsKeyDown(VK_OEM_6)) // ']' rotates it the other way
                {
                    m_polarizerAngleRadians += m_config.polarizerRotateSpeed * deltaTime;
                }
            }
        }

        if (!windowFocused)
        {
            return;
        }

        const float maxPitch = GetCameraMaxPitchRadians();
        if (IsKeyDown(VK_LEFT))
        {
            m_cameraYaw -= m_config.keyLookSpeed * deltaTime;
        }
        if (IsKeyDown(VK_RIGHT))
        {
            m_cameraYaw += m_config.keyLookSpeed * deltaTime;
        }
        if (IsKeyDown(VK_UP))
        {
            m_cameraPitch += m_config.keyLookSpeed * deltaTime;
        }
        if (IsKeyDown(VK_DOWN))
        {
            m_cameraPitch -= m_config.keyLookSpeed * deltaTime;
        }
        m_cameraPitch = std::clamp(m_cameraPitch, -maxPitch, maxPitch);
    }

    // Register the window class and create the main render window at
    // the configured resolution. Returns once the window is visible.
    void CreateWindowAndShow()
    {
        const HINSTANCE instance = GetModuleHandleW(nullptr);
        const wchar_t* className = L"VulkanPathTracerWindowClass";

        WNDCLASSEXW windowClass{};
        windowClass.cbSize = sizeof(windowClass);
        windowClass.hInstance = instance;
        windowClass.lpfnWndProc = WindowProc;
        windowClass.lpszClassName = className;
        windowClass.hCursor = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));

        const ATOM atom = RegisterClassExW(&windowClass);
        if (atom == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
        {
            throw std::runtime_error("Failed to register window class.");
        }

        RECT rect{0, 0, static_cast<LONG>(m_config.width), static_cast<LONG>(m_config.height)};
        const DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
        ThrowIfFalse(AdjustWindowRect(&rect, style, FALSE), "Failed to size window.");

        m_window = CreateWindowExW(0,
                                   className,
                                   L"Vulkan Path Tracer - FPS: measuring...",
                                   style,
                                   CW_USEDEFAULT,
                                   CW_USEDEFAULT,
                                   rect.right - rect.left,
                                   rect.bottom - rect.top,
                                   nullptr,
                                   nullptr,
                                   instance,
                                   nullptr);
        if (m_window == nullptr)
        {
            throw std::runtime_error("Failed to create window.");
        }

        ShowWindow(m_window, SW_SHOWDEFAULT);
        UpdateWindow(m_window);
        std::printf("[Config] Edit %s and save to hot-reload tuning.\n", m_configPath.string().c_str());
        std::puts("[Config] width, height, and frameCount are loaded from JSON at startup.");
        std::puts("[Controls] Hold RMB or use the arrow keys to look around the sky. R resets the view.");
        std::puts("[Controls] P toggles the polarization filter; C switches linear/elliptical.");
        std::puts("[Controls] Linear: [ ] rotate the filter axis. Elliptical: [ ] adjust ellipticity.");
    }

    // Create the VkInstance via vk-bootstrap, which auto-enables the Win32
    // surface extensions. Validation layers are requested only when present so
    // the SDK's debug layers light up without a hard dependency on them.
    void CreateInstance()
    {
        auto instanceResult = vkb::InstanceBuilder{}
                                  .set_app_name("Vulkan Path Tracer")
                                  .set_engine_name("None")
                                  .require_api_version(1, 2, 0)
                                  .request_validation_layers()
                                  .build();
        if (!instanceResult)
        {
            throw std::runtime_error("Failed to create Vulkan instance: " + instanceResult.error().message());
        }

        m_vkbInstance = instanceResult.value();
        m_instance = vk::raii::Instance(m_context, m_vkbInstance.instance);
    }

    // Create the Win32 surface linking the HWND to the Vulkan instance.
    void CreateSurface()
    {
        vk::Win32SurfaceCreateInfoKHR createInfo{};
        createInfo.hinstance = GetModuleHandleW(nullptr);
        createInfo.hwnd = m_window;
        m_surface = m_instance.createWin32SurfaceKHR(createInfo);
    }

    // Select a physical device via vk-bootstrap. vkb requires a presentable
    // swapchain by default; we additionally require the swapchain surface to
    // support storage-image usage (the compute shader writes it directly) and
    // the non-semantic-info extension that keeps debugPrintfEXT working.
    void PickPhysicalDevice()
    {
        VkPhysicalDeviceFeatures requiredFeatures{};
        requiredFeatures.shaderStorageImageWriteWithoutFormat = VK_TRUE;

        auto deviceResult = vkb::PhysicalDeviceSelector{m_vkbInstance}
                                .set_surface(static_cast<VkSurfaceKHR>(*m_surface))
                                .set_minimum_version(1, 2)
                                .set_required_features(requiredFeatures)
                                .add_required_extension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME)
                                .require_present()
                                .select();
        if (!deviceResult)
        {
            throw std::runtime_error("Failed to select Vulkan physical device: " + deviceResult.error().message());
        }

        m_vkbPhysicalDevice = deviceResult.value();
        m_physicalDevice = vk::raii::PhysicalDevice(m_instance, m_vkbPhysicalDevice.physical_device);

        const vk::SurfaceCapabilitiesKHR capabilities = m_physicalDevice.getSurfaceCapabilitiesKHR(*m_surface);
        if (!(capabilities.supportedUsageFlags & vk::ImageUsageFlagBits::eStorage))
        {
            throw std::runtime_error("Selected GPU does not support storage-image swapchains for this app.");
        }
    }

    // Create the VkDevice via vk-bootstrap and fetch the graphics+present
    // queue handles and their family indices. vkb already enabled the
    // swapchain and non-semantic-info extensions during device selection.
    void CreateLogicalDevice()
    {
        auto deviceResult = vkb::DeviceBuilder{m_vkbPhysicalDevice}.build();
        if (!deviceResult)
        {
            throw std::runtime_error("Failed to create Vulkan logical device: " + deviceResult.error().message());
        }

        m_vkbDevice = deviceResult.value();
        m_device = vk::raii::Device(m_physicalDevice, m_vkbDevice.device);

        auto graphicsFamily = m_vkbDevice.get_queue_index(vkb::QueueType::graphics);
        auto presentFamily = m_vkbDevice.get_queue_index(vkb::QueueType::present);
        if (!graphicsFamily || !presentFamily)
        {
            throw std::runtime_error("Failed to retrieve graphics/present queue families.");
        }
        m_queueFamilies.graphicsFamily = graphicsFamily.value();
        m_queueFamilies.presentFamily = presentFamily.value();
        m_graphicsQueue = m_device.getQueue(graphicsFamily.value(), 0);
        m_presentQueue = m_device.getQueue(presentFamily.value(), 0);
    }

    // Create the VMA allocator that backs every device buffer allocation.
    void CreateAllocator()
    {
        VmaAllocatorCreateInfo allocatorInfo{};
        allocatorInfo.instance = static_cast<VkInstance>(*m_instance);
        allocatorInfo.physicalDevice = static_cast<VkPhysicalDevice>(*m_physicalDevice);
        allocatorInfo.device = static_cast<VkDevice>(*m_device);
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;
        ThrowVk(vmaCreateAllocator(&allocatorInfo, &m_allocator), "Failed to create VMA allocator");
    }

    // Allocate a host-visible, persistently mapped buffer through VMA. Every
    // buffer in this renderer is a small CPU-written upload target (the scene
    // UBO and the baked Mie SSBO), so the allocation strategy is shared.
    BufferAllocation CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage) const
    {
        BufferAllocation allocation{};
        allocation.size = size;

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocCreateInfo{};
        allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                                | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo info{};
        ThrowVk(vmaCreateBuffer(m_allocator, &bufferInfo, &allocCreateInfo, &allocation.buffer, &allocation.allocation, &info),
                "Failed to create buffer");
        allocation.mapped = info.pMappedData;
        return allocation;
    }

    // Destroy the VkBuffer and free its VMA allocation, zeroing the handle so
    // the struct can be safely re-used or destroyed twice.
    void DestroyBuffer(BufferAllocation& allocation)
    {
        if (allocation.buffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(m_allocator, allocation.buffer, allocation.allocation);
            allocation.buffer = VK_NULL_HANDLE;
            allocation.allocation = VK_NULL_HANDLE;
        }
        allocation.mapped = nullptr;
        allocation.size = 0;
    }

    // Copy host data into the buffer's persistently mapped allocation, then
    // flush so the write is visible to the GPU even on non-coherent memory.
    // The std::span overload lets callers pass any trivially-copyable object or
    // contiguous range via std::as_bytes without hand-computing byte sizes.
    void UploadToBuffer(const BufferAllocation& allocation, std::span<const std::byte> data) const
    {
        if (data.size() > static_cast<size_t>(allocation.size))
        {
            throw std::runtime_error("Upload exceeds destination buffer size.");
        }
        std::memcpy(allocation.mapped, data.data(), data.size());
        ThrowVk(vmaFlushAllocation(m_allocator, allocation.allocation, 0, data.size()),
                "Failed to flush buffer allocation");
    }

    // Allocate the scene UBO + Mie SSBO and upload the sky parameters. There
    // is no geometry / acceleration structure in this sky-only renderer.
    void CreateSceneResources()
    {
        CreateSceneBuffers();
        UploadSceneDataFromConfig();
    }

    // Create the swapchain and its image views via vk-bootstrap. The images
    // are created with VK_IMAGE_USAGE_STORAGE_BIT and written directly by the
    // compute shader (no separate offscreen image or blit). Present-mode
    // priority is IMMEDIATE → MAILBOX → FIFO (FIFO is the guaranteed fallback);
    // at least frameCount images are requested so per-frame resources line up.
    void CreateSwapchain()
    {
        auto swapchainResult = vkb::SwapchainBuilder{m_vkbDevice}
                                   .set_desired_format({VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
                                   .set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
                                   .set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)
                                   .set_desired_extent(m_config.width, m_config.height)
                                   .set_image_usage_flags(VK_IMAGE_USAGE_STORAGE_BIT)
                                   .set_required_min_image_count(m_config.frameCount)
                                   .build();
        if (!swapchainResult)
        {
            throw std::runtime_error("Failed to create Vulkan swapchain: " + swapchainResult.error().message());
        }

        vkb::Swapchain swapchain = swapchainResult.value();
        m_swapchain = vk::raii::SwapchainKHR(m_device, swapchain.swapchain);
        m_swapchainFormat = static_cast<vk::Format>(swapchain.image_format);
        m_swapchainExtent = swapchain.extent;

        // Wrap the swapchain images and build a 2D color image view per image.
        m_swapchainImages.clear();
        m_swapchainImageViews.clear();
        for (VkImage image : m_swapchain.getImages())
        {
            m_swapchainImages.emplace_back(image);

            vk::ImageViewCreateInfo viewInfo{};
            viewInfo.image = image;
            viewInfo.viewType = vk::ImageViewType::e2D;
            viewInfo.format = m_swapchainFormat;
            viewInfo.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            m_swapchainImageViews.emplace_back(m_device, viewInfo);
        }
        m_swapchainLayouts.assign(m_swapchainImages.size(), vk::ImageLayout::eUndefined);
    }

    // Define the descriptor-set layout used by the compute pipeline: the
    // output storage image (binding 0), the sky parameter UBO (binding 2),
    // and the baked Lorenz–Mie SSBO (binding 7).
    void CreateDescriptorSetLayout()
    {
        // Output image (binding 0), sky parameter UBO (binding 2), and the
        // Lorenz–Mie scattering-matrix SSBO (binding 7) — all read/written by
        // the ray-generation shader. The binding numbers keep their original
        // values (gaps are legal) so the shared sky header is untouched.
        using enum vk::DescriptorType;
        constexpr auto compute = vk::ShaderStageFlagBits::eCompute;
        const std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {
            vk::DescriptorSetLayoutBinding{0, eStorageImage, 1, compute},
            vk::DescriptorSetLayoutBinding{2, eUniformBuffer, 1, compute},
            vk::DescriptorSetLayoutBinding{7, eStorageBuffer, 1, compute},
        };

        vk::DescriptorSetLayoutCreateInfo createInfo{};
        createInfo.setBindings(bindings);
        m_descriptorSetLayout = vk::raii::DescriptorSetLayout(m_device, createInfo);
    }

    vk::raii::ShaderModule CreateShaderModule(const std::vector<char>& bytecode)
    {
        vk::ShaderModuleCreateInfo createInfo{};
        createInfo.codeSize = bytecode.size();
        createInfo.pCode = std::bit_cast<const uint32_t*>(bytecode.data());
        return vk::raii::ShaderModule(m_device, createInfo);
    }

    // Compile the compute shader module from its SPIR-V blob and assemble the
    // pipeline. Sky-only renderer: a single compute shader evaluates the sky
    // analytically per pixel and writes the swapchain storage image directly.
    void CreatePipeline()
    {
        const vk::raii::ShaderModule computeModule = CreateShaderModule(LoadBinaryFile(L"path_tracer.comp.spv"));

        vk::PipelineShaderStageCreateInfo stageInfo{};
        stageInfo.stage = vk::ShaderStageFlagBits::eCompute;
        stageInfo.module = *computeModule;
        stageInfo.pName = "main";

        const vk::PushConstantRange pushRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants)};
        vk::PipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.setSetLayouts(*m_descriptorSetLayout);
        layoutInfo.setPushConstantRanges(pushRange);
        m_pipelineLayout = vk::raii::PipelineLayout(m_device, layoutInfo);

        vk::ComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = *m_pipelineLayout;
        m_computePipeline = vk::raii::Pipeline(m_device, nullptr, pipelineInfo);
    }

    // Allocate the descriptor pool and one descriptor set per layout
    // slot. Initial bindings are deferred to UpdateDescriptorSetContents()
    // because they depend on scene resources that are created later.
    void CreateDescriptorSets()
    {
        using enum vk::DescriptorType;
        const uint32_t count = static_cast<uint32_t>(m_swapchainImageViews.size());
        const std::array<vk::DescriptorPoolSize, 3> poolSizes = {
            vk::DescriptorPoolSize{eStorageImage, count},
            vk::DescriptorPoolSize{eUniformBuffer, count},
            vk::DescriptorPoolSize{eStorageBuffer, count},
        };

        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        poolInfo.maxSets = count;
        poolInfo.setPoolSizes(poolSizes);
        m_descriptorPool = vk::raii::DescriptorPool(m_device, poolInfo);

        const std::vector<vk::DescriptorSetLayout> layouts(count, *m_descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo{};
        allocInfo.descriptorPool = *m_descriptorPool;
        allocInfo.setSetLayouts(layouts);

        m_descriptorSets = vk::raii::DescriptorSets(m_device, allocInfo);
        UpdateDescriptorSetContents();
    }

    // Create the graphics-queue command pool used for the per-frame command
    // buffers.
    void CreateCommandPool()
    {
        vk::CommandPoolCreateInfo createInfo{vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                             m_queueFamilies.graphicsFamily.value()};
        m_commandPool = vk::raii::CommandPool(m_device, createInfo);
    }

    // Allocate one primary command buffer per frame in flight.
    void CreateCommandBuffers()
    {
        vk::CommandBufferAllocateInfo allocInfo{*m_commandPool, vk::CommandBufferLevel::ePrimary, m_config.frameCount};
        m_commandBuffers = vk::raii::CommandBuffers(m_device, allocInfo);
    }

    // Allocate the per-frame semaphores (image-available, render-
    // finished) and fences that gate command-buffer reuse.
    void CreateSyncObjects()
    {
        const vk::FenceCreateInfo fenceInfo{vk::FenceCreateFlagBits::eSignaled};

        m_frames.clear();
        for (uint32_t i = 0; i < m_config.frameCount; ++i)
        {
            FrameResources frame{
                m_device.createSemaphore({}),
                m_device.createSemaphore({}),
                m_device.createFence(fenceInfo),
            };
            m_frames.push_back(std::move(frame));
        }
    }

    PushConstants BuildPushConstants() const
    {
        PushConstants constants{};
        constants.cameraPositionFrame[0] = m_cameraPosition.x;
        constants.cameraPositionFrame[1] = m_cameraPosition.y;
        constants.cameraPositionFrame[2] = m_cameraPosition.z;
        constants.cameraPositionFrame[3] = static_cast<float>(m_frameIndex);

        const Vec3 forward = GetCameraForward();
        const Vec3 right = Normalize(Cross({0.0f, 1.0f, 0.0f}, forward));
        const Vec3 up = Normalize(Cross(forward, right));

        constants.cameraForwardSamples[0] = forward.x;
        constants.cameraForwardSamples[1] = forward.y;
        constants.cameraForwardSamples[2] = forward.z;
        constants.cameraForwardSamples[3] = static_cast<float>(m_config.samplesPerPixel);
        constants.cameraRightBounces[0] = right.x;
        constants.cameraRightBounces[1] = right.y;
        constants.cameraRightBounces[2] = right.z;
        constants.cameraRightBounces[3] = 0.0f; // (was max bounces; unused in the sky-only renderer)
        constants.cameraUpTanHalfFovY[0] = up.x;
        constants.cameraUpTanHalfFovY[1] = up.y;
        constants.cameraUpTanHalfFovY[2] = up.z;
        constants.cameraUpTanHalfFovY[3] = std::tan(m_config.fovYDegrees * 0.5f * kPi / 180.0f);
        constants.skyBottomExposure[0] = m_config.skyBottomColor[0];
        constants.skyBottomExposure[1] = m_config.skyBottomColor[1];
        constants.skyBottomExposure[2] = m_config.skyBottomColor[2];
        constants.skyBottomExposure[3] = m_config.skyExposure;
        constants.skyTopAspect[0] = m_config.skyTopColor[0];
        constants.skyTopAspect[1] = m_config.skyTopColor[1];
        constants.skyTopAspect[2] = m_config.skyTopColor[2];
        constants.skyTopAspect[3] =
            static_cast<float>(m_swapchainExtent.width) / static_cast<float>(m_swapchainExtent.height);

        constants.polarizer[0] = m_polarizerEnabled ? 1.0f : 0.0f;
        constants.polarizer[1] = m_polarizerAngleRadians;
        constants.polarizer[2] = m_polarizerElliptical ? m_polarizerEllipticityRadians : 0.0f;
        constants.polarizer[3] = 0.0f;

        constants.imageSize[0] = m_swapchainExtent.width;
        constants.imageSize[1] = m_swapchainExtent.height;
        return constants;
    }

    // Record one frame's worth of work into commandBuffer:
    //   1. Barrier transitioning the acquired swapchain image to GENERAL.
    //   2. Bind the compute pipeline + descriptors, push constants.
    //   3. vkCmdDispatch over an 8x8-tiled grid; the shader writes the
    //      swapchain image directly as a storage image.
    //   4. Barrier transitioning the swapchain image to PRESENT_SRC.
    void RecordCommandBuffer(const vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
    {
        commandBuffer.begin({});

        const vk::ImageSubresourceRange colorRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        vk::ImageMemoryBarrier toGeneral{};
        toGeneral.oldLayout = m_swapchainLayouts[imageIndex];
        toGeneral.newLayout = vk::ImageLayout::eGeneral;
        toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.image = m_swapchainImages[imageIndex];
        toGeneral.subresourceRange = colorRange;
        toGeneral.srcAccessMask = {};
        toGeneral.dstAccessMask = vk::AccessFlagBits::eShaderWrite;

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                      vk::PipelineStageFlagBits::eComputeShader,
                                      {}, nullptr, nullptr, toGeneral);

        const PushConstants pushConstants = BuildPushConstants();
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *m_computePipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                         *m_pipelineLayout,
                                         0,
                                         *m_descriptorSets[imageIndex],
                                         nullptr);
        commandBuffer.pushConstants<PushConstants>(*m_pipelineLayout,
                                                   vk::ShaderStageFlagBits::eCompute,
                                                   0,
                                                   pushConstants);
        // 8x8 workgroups, matching the compute shader's local size; round up so
        // the whole render target is covered (the shader discards the overhang).
        constexpr uint32_t kTile = 8;
        const uint32_t groupsX = (m_swapchainExtent.width + kTile - 1) / kTile;
        const uint32_t groupsY = (m_swapchainExtent.height + kTile - 1) / kTile;
        commandBuffer.dispatch(groupsX, groupsY, 1);

        vk::ImageMemoryBarrier toPresent = toGeneral;
        toPresent.oldLayout = vk::ImageLayout::eGeneral;
        toPresent.newLayout = vk::ImageLayout::ePresentSrcKHR;
        toPresent.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        toPresent.dstAccessMask = {};

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                      vk::PipelineStageFlagBits::eBottomOfPipe,
                                      {}, nullptr, nullptr, toPresent);

        commandBuffer.end();
        m_swapchainLayouts[imageIndex] = vk::ImageLayout::ePresentSrcKHR;
    }

    // Wait for the current in-flight slot, acquire a swapchain image,
    // record + submit the frame's command buffer, then present.
    // Handles VK_ERROR_OUT_OF_DATE_KHR by recreating the swapchain.
    void RenderFrame()
    {
        FrameResources& frame = m_frames[m_currentFrame];
        while (m_device.waitForFences(*frame.inFlight, VK_TRUE, UINT64_MAX) == vk::Result::eTimeout)
        {
        }

        const auto [acquireResult, imageIndex] = m_swapchain.acquireNextImage(UINT64_MAX, *frame.imageAvailable);
        if (acquireResult != vk::Result::eSuccess && acquireResult != vk::Result::eSuboptimalKHR)
        {
            throw std::runtime_error("Failed to acquire swapchain image.");
        }

        m_device.resetFences(*frame.inFlight);
        const vk::raii::CommandBuffer& commandBuffer = m_commandBuffers[m_currentFrame];
        commandBuffer.reset();
        RecordCommandBuffer(commandBuffer, imageIndex);

        const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eComputeShader;
        vk::SubmitInfo submitInfo{};
        submitInfo.setWaitSemaphores(*frame.imageAvailable);
        submitInfo.setWaitDstStageMask(waitStage);
        submitInfo.setCommandBuffers(*commandBuffer);
        submitInfo.setSignalSemaphores(*frame.renderFinished);
        m_graphicsQueue.submit(submitInfo, *frame.inFlight);

        vk::PresentInfoKHR presentInfo{};
        presentInfo.setWaitSemaphores(*frame.renderFinished);
        presentInfo.setSwapchains(*m_swapchain);
        presentInfo.setImageIndices(imageIndex);
        const vk::Result present = m_presentQueue.presentKHR(presentInfo);
        if (present != vk::Result::eSuccess && present != vk::Result::eSuboptimalKHR)
        {
            throw std::runtime_error("Failed to present swapchain image.");
        }

        ++m_frameIndex;
        m_currentFrame = (m_currentFrame + 1) % static_cast<uint32_t>(m_frames.size());
    }

    // Throttled window-title update displaying live FPS and per-frame
    // milliseconds — cheap diagnostic for performance tuning.
    void UpdateWindowTitle(double fps, double frameMs)
    {
        const std::wstring title = std::format(L"Vulkan Path Tracer - {:.1f} FPS ({:.2f} ms)", fps, frameMs);
        SetWindowTextW(m_window, title.c_str());
    }

    // Main loop: pump Win32 messages, hot-reload config, advance camera,
    // and render until WM_QUIT. Uses a fixed-timestep clock for the
    // camera integration so movement speed is independent of frame
    // rate.
    void MessageLoop()
    {
        using Clock = std::chrono::steady_clock;
        MSG message{};
        auto statsStart = Clock::now();
        auto previousFrameStart = statsStart;
        uint32_t framesSinceUpdate = 0;

        while (true)
        {
            while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
            {
                if (message.message == WM_QUIT)
                {
                    return;
                }

                TranslateMessage(&message);
                DispatchMessageW(&message);
            }

            const auto frameStart = Clock::now();
            ReloadRuntimeConfigIfNeeded();
            const double deltaSeconds = std::chrono::duration<double>(frameStart - previousFrameStart).count();
            previousFrameStart = frameStart;
            UpdateCamera(deltaSeconds);
            RenderFrame();
            const auto frameEnd = Clock::now();

            ++framesSinceUpdate;
            const double elapsedSeconds = std::chrono::duration<double>(frameEnd - statsStart).count();
            if (elapsedSeconds >= 1.0)
            {
                const double fps = static_cast<double>(framesSinceUpdate) / elapsedSeconds;
                const double frameMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
                UpdateWindowTitle(fps, frameMs);
                std::printf("[Vulkan] %.1f FPS (%.2f ms)\n", fps, frameMs);
                framesSinceUpdate = 0;
                statsStart = frameEnd;
            }
        }
    }

    struct FrameResources
    {
        vk::raii::Semaphore imageAvailable{nullptr};
        vk::raii::Semaphore renderFinished{nullptr};
        vk::raii::Fence inFlight{nullptr};
    };

    HWND m_window = nullptr;

    // vk-bootstrap wrappers retained for the lifetime of the app: they own the
    // builder-side metadata used to fetch queue family indices.
    vkb::Instance m_vkbInstance;
    vkb::PhysicalDevice m_vkbPhysicalDevice;
    vkb::Device m_vkbDevice;

    // RAII Vulkan handles. Declaration order is destruction-reverse order:
    // device-child objects are declared after the device so they are destroyed
    // before it; the device after the instance; the instance last.
    vk::raii::Context m_context;
    vk::raii::Instance m_instance{nullptr};
    vk::raii::SurfaceKHR m_surface{nullptr};
    vk::raii::PhysicalDevice m_physicalDevice{nullptr};
    vk::raii::Device m_device{nullptr};
    vk::raii::Queue m_graphicsQueue{nullptr};
    vk::raii::Queue m_presentQueue{nullptr};
    QueueFamilyIndices m_queueFamilies;

    // VMA allocator + its buffers are released manually in the destructor body
    // (before the raii members, while the device is still alive).
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    BufferAllocation m_sceneDataBuffer{};
    BufferAllocation m_mieScatteringBuffer{};

    vk::raii::SwapchainKHR m_swapchain{nullptr};
    vk::Format m_swapchainFormat = vk::Format::eUndefined;
    vk::Extent2D m_swapchainExtent{};
    std::vector<vk::Image> m_swapchainImages;
    std::vector<vk::raii::ImageView> m_swapchainImageViews;
    std::vector<vk::ImageLayout> m_swapchainLayouts;

    vk::raii::DescriptorSetLayout m_descriptorSetLayout{nullptr};
    vk::raii::PipelineLayout m_pipelineLayout{nullptr};
    vk::raii::Pipeline m_computePipeline{nullptr};
    vk::raii::DescriptorPool m_descriptorPool{nullptr};
    std::vector<vk::raii::DescriptorSet> m_descriptorSets;

    vk::raii::CommandPool m_commandPool{nullptr};
    std::vector<vk::raii::CommandBuffer> m_commandBuffers;
    std::vector<FrameResources> m_frames;
    uint32_t m_currentFrame = 0;
    uint64_t m_frameIndex = 0;
    RuntimeConfig m_config{};
    std::filesystem::path m_configPath;
    std::filesystem::file_time_type m_configLastWriteTime{};
    std::chrono::steady_clock::time_point m_lastConfigPollTime{};
    Vec3 m_cameraPosition{};
    float m_cameraYaw = 0.0f;
    float m_cameraPitch = 0.0f;
    bool m_resetCameraKeyDown = false;
    bool m_mouseLookActive = false;
    POINT m_lastMousePosition{};

    // Camera polarization filter. P toggles it on/off; C switches between a
    // linear analyzer and an elliptical one. In linear mode [ and ] rotate the
    // major axis (measured in the image plane from camera-right, positive
    // toward camera-down on screen — counterclockwise facing the incoming
    // beam); in elliptical mode [ and ] adjust ellipticity (-45..+45 degrees).
    bool m_polarizerEnabled = false;
    float m_polarizerAngleRadians = 0.0f;
    bool m_polarizerToggleKeyDown = false;
    bool m_polarizerElliptical = false;
    bool m_polarizerModeKeyDown = false;
    float m_polarizerEllipticityRadians = kPi * 0.125f;
};

// Public C-style entry point exported by VulkanPathTracer.h. Constructs a
// VulkanPathTracer instance on the stack and runs it; any throw escapes
// upward to main().
void RunVulkanPathTracer()
{
    VulkanPathTracer app;
    app.Run();
}
