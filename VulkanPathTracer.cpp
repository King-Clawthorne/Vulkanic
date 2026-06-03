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
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <optional>
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
        char buffer[256]{};
        std::snprintf(buffer, sizeof(buffer), "%s (VkResult=%d)", message, static_cast<int>(result));
        throw std::runtime_error(buffer);
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

// Snapshot of what a (physical device, surface) pair supports. Captured
// once during init and again on swapchain recreation (window resize).
struct SwapchainSupport
{
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

// Generic GPU buffer + its backing memory. Owned together so destruction
// is just a matched pair of vk*Free / vkDestroyBuffer calls.
struct BufferAllocation
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

// Sky / scene uniforms uploaded to the compute pipeline.
//
// Each vec4 packs multiple scalars together to keep the std140 layout
// compact and to match the shader-side struct one-for-one. The static
// assertion below pins the size at 96 bytes so any future field reshuffle
// that breaks alignment fails to compile rather than corrupting the GPU
// view of the data.
struct alignas(16) SceneData
{
    float skyBetaRayleighBetaM[4];
    float skyMieEarthAtmosScaleHr[4];
    float skyScaleHmSunRadiusAa[4];
    float skySunRadiance[4];
    float skySunDirection[4];
    uint32_t skySampleCounts[4];
    // Vector radiative transfer: x = Rayleigh depolarization, y = scattering
    // orders, z = Mie table angle bins, w unused. The Mie scattering matrix
    // itself rides in the binding-7 SSBO, not here.
    float skyVrtParams[4];
};

static_assert(sizeof(SceneData) == 112, "Scene data layout must stay 16-byte aligned.");

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

    ~VulkanPathTracer()
    {
        Cleanup();
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
        // Command pool is created before scene setup because the Mie SSBO upload uses one-time command buffers.
        CreateCommandPool();
        CreateSceneResources();
        CreateSwapchain();
        CreateDescriptorSetLayout();
        CreatePipeline();
        CreateDescriptorSets();
        CreateCommandBuffers();
        CreateSyncObjects();
        MessageLoop();
        vkDeviceWaitIdle(m_device);
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
        SceneData sceneData{};
        sceneData.skyBetaRayleighBetaM[0] = m_config.skySpectral.betaRayleigh[0];
        sceneData.skyBetaRayleighBetaM[1] = m_config.skySpectral.betaRayleigh[1];
        sceneData.skyBetaRayleighBetaM[2] = m_config.skySpectral.betaRayleigh[2];
        sceneData.skyBetaRayleighBetaM[3] = m_config.skySpectral.betaMie;
        sceneData.skyMieEarthAtmosScaleHr[0] = m_config.skySpectral.mieG;
        sceneData.skyMieEarthAtmosScaleHr[1] = m_config.skySpectral.earthRadius;
        sceneData.skyMieEarthAtmosScaleHr[2] = m_config.skySpectral.atmosphereRadius;
        sceneData.skyMieEarthAtmosScaleHr[3] = m_config.skySpectral.scaleHeightRayleigh;
        sceneData.skyScaleHmSunRadiusAa[0] = m_config.skySpectral.scaleHeightMie;
        sceneData.skyScaleHmSunRadiusAa[1] = m_config.skySpectral.sunRadius;
        sceneData.skyScaleHmSunRadiusAa[2] = m_config.skySpectral.sunAa;
        sceneData.skySunRadiance[0] = m_config.skySpectral.sunRadiance[0];
        sceneData.skySunRadiance[1] = m_config.skySpectral.sunRadiance[1];
        sceneData.skySunRadiance[2] = m_config.skySpectral.sunRadiance[2];
        sceneData.skySunDirection[0] = m_config.skySpectral.sunDirection[0];
        sceneData.skySunDirection[1] = m_config.skySpectral.sunDirection[1];
        sceneData.skySunDirection[2] = m_config.skySpectral.sunDirection[2];
        sceneData.skySampleCounts[0] = m_config.skySpectral.secondarySamples;
        sceneData.skySampleCounts[1] = m_config.skySpectral.viewSteps;
        sceneData.skySampleCounts[2] = m_config.skySpectral.samples;
        sceneData.skyVrtParams[0] = m_config.skySpectral.rayleighDepolarization;
        sceneData.skyVrtParams[1] = 0.0f; // (was scattering orders; multiple scattering removed)
        sceneData.skyVrtParams[2] = static_cast<float>(m_config.skySpectral.mieTableAngleBins);
        sceneData.skyVrtParams[3] = 0.0f;
        return sceneData;
    }

    // Allocate the scene uniform buffer (sky parameters) and the Lorenz–Mie
    // scattering-matrix SSBO consumed by the ray-generation shader.
    void CreateSceneBuffers()
    {
        m_sceneDataBuffer = CreateBuffer(sizeof(SceneData),
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         false);
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
        m_mieScatteringBuffer = CreateBuffer(size,
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                             false);
        UploadToBuffer(m_mieScatteringBuffer, table.data(), table.size() * sizeof(MieMatrixEntry));
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
        UploadToBuffer(m_sceneDataBuffer, &sceneData, sizeof(sceneData));
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
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageView = m_swapchainImageViews[i];
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = m_descriptorSets[i];
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write.pImageInfo = &imageInfo;

            VkDescriptorBufferInfo sceneDataInfo{};
            sceneDataInfo.buffer = m_sceneDataBuffer.buffer;
            sceneDataInfo.offset = 0;
            sceneDataInfo.range = m_sceneDataBuffer.size;

            VkWriteDescriptorSet sceneWrite{};
            sceneWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            sceneWrite.dstSet = m_descriptorSets[i];
            sceneWrite.dstBinding = 2;
            sceneWrite.descriptorCount = 1;
            sceneWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            sceneWrite.pBufferInfo = &sceneDataInfo;

            VkDescriptorBufferInfo mieDataInfo{};
            mieDataInfo.buffer = m_mieScatteringBuffer.buffer;
            mieDataInfo.offset = 0;
            mieDataInfo.range = m_mieScatteringBuffer.size;

            VkWriteDescriptorSet mieWrite{};
            mieWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            mieWrite.dstSet = m_descriptorSets[i];
            mieWrite.dstBinding = 7;
            mieWrite.descriptorCount = 1;
            mieWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            mieWrite.pBufferInfo = &mieDataInfo;

            const std::array<VkWriteDescriptorSet, 3> writes = {
                write,
                sceneWrite,
                mieWrite,
            };
            vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
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

        ThrowVk(vkDeviceWaitIdle(m_device), "Failed to wait for device idle during config reload");
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

    // Create the VkInstance with the platform surface extensions
    // required for Win32. No validation layers are requested by default;
    // they can be enabled via the SDK's environment variables.
    void CreateInstance()
    {
        const std::array<const char*, 2> extensions = {
            VK_KHR_SURFACE_EXTENSION_NAME,
            VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        };

        VkApplicationInfo applicationInfo{};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "Vulkan Path Tracer";
        applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        applicationInfo.pEngineName = "None";
        applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        applicationInfo.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &applicationInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        ThrowVk(vkCreateInstance(&createInfo, nullptr, &m_instance), "Failed to create Vulkan instance");
    }

    // Create the Win32 VkSurfaceKHR linking the HWND to the Vulkan
    // instance.
    void CreateSurface()
    {
        VkWin32SurfaceCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        createInfo.hinstance = GetModuleHandleW(nullptr);
        createInfo.hwnd = m_window;
        ThrowVk(vkCreateWin32SurfaceKHR(m_instance, &createInfo, nullptr, &m_surface),
                "Failed to create Win32 surface");
    }

    QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) const
    {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, families.data());

        for (uint32_t i = 0; i < queueFamilyCount; ++i)
        {
            if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
            {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &presentSupported);
            if (presentSupported == VK_TRUE)
            {
                indices.presentFamily = i;
            }

            if (indices.IsComplete())
            {
                break;
            }
        }

        return indices;
    }

    SwapchainSupport QuerySwapchainSupport(VkPhysicalDevice device) const
    {
        SwapchainSupport support{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, m_surface, &support.capabilities);

        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, nullptr);
        support.formats.resize(formatCount);
        if (formatCount > 0)
        {
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, support.formats.data());
        }

        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &presentModeCount, nullptr);
        support.presentModes.resize(presentModeCount);
        if (presentModeCount > 0)
        {
            vkGetPhysicalDeviceSurfacePresentModesKHR(device,
                                                      m_surface,
                                                      &presentModeCount,
                                                      support.presentModes.data());
        }

        return support;
    }

    bool IsDeviceSuitable(VkPhysicalDevice device)
    {
        const auto queueFamilies = FindQueueFamilies(device);
        if (!queueFamilies.IsComplete())
        {
            return false;
        }

        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());

        // Compute-only renderer: a presentable swapchain is the only hard
        // requirement (compute itself is core Vulkan).
        bool hasSwapchain = false;
        for (const auto& extension : extensions)
        {
            if (std::strcmp(extension.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0)
            {
                hasSwapchain = true;
                break;
            }
        }
        if (!hasSwapchain)
        {
            return false;
        }

        const auto swapchainSupport = QuerySwapchainSupport(device);
        if (swapchainSupport.formats.empty() || swapchainSupport.presentModes.empty())
        {
            return false;
        }

        return (swapchainSupport.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT) != 0;
    }

    // Iterate adapters and select the first one that supports the required
    // queue families and a storage-image swapchain. Throws if none qualifies.
    void PickPhysicalDevice()
    {
        uint32_t deviceCount = 0;
        ThrowVk(vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr),
                "Failed to enumerate Vulkan physical devices");
        if (deviceCount == 0)
        {
            throw std::runtime_error("No Vulkan-capable GPU found.");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        ThrowVk(vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data()),
                "Failed to enumerate Vulkan physical devices");

        for (VkPhysicalDevice device : devices)
        {
            if (IsDeviceSuitable(device))
            {
                m_physicalDevice = device;
                m_queueFamilies = FindQueueFamilies(device);
                break;
            }
        }

        if (m_physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("No Vulkan device supports storage-image swapchains for this app.");
        }
    }

    // Create the VkDevice (swapchain extension only) and fetch the
    // graphics+present queue handles.
    void CreateLogicalDevice()
    {
        const float queuePriority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queueInfos;
        const std::array<uint32_t, 2> uniqueFamilies = {
            m_queueFamilies.graphicsFamily.value(),
            m_queueFamilies.presentFamily.value(),
        };

        std::vector<uint32_t> familyList;
        for (uint32_t family : uniqueFamilies)
        {
            if (std::find(familyList.begin(), familyList.end(), family) == familyList.end())
            {
                familyList.push_back(family);
            }
        }

        for (uint32_t family : familyList)
        {
            VkDeviceQueueCreateInfo queueInfo{};
            queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueInfo.queueFamilyIndex = family;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &queuePriority;
            queueInfos.push_back(queueInfo);
        }

        // Compute-only renderer: a swapchain is the only device extension we
        // need. (Shader non-semantic info keeps debugPrintfEXT working under the
        // -g shader build.)
        const std::array<const char*, 2> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME
        };

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
        createInfo.pQueueCreateInfos = queueInfos.data();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        ThrowVk(vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device),
                "Failed to create Vulkan logical device");

        vkGetDeviceQueue(m_device, m_queueFamilies.graphicsFamily.value(), 0, &m_graphicsQueue);
        vkGetDeviceQueue(m_device, m_queueFamilies.presentFamily.value(), 0, &m_presentQueue);
    }

    uint32_t FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties) const
    {
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
        {
            if ((typeBits & (1u << i)) != 0
                && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable Vulkan memory type.");
    }

    BufferAllocation CreateBuffer(VkDeviceSize size,
                                  VkBufferUsageFlags usage,
                                  VkMemoryPropertyFlags properties,
                                  bool enableDeviceAddress) const
    {
        BufferAllocation allocation{};
        allocation.size = size;

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ThrowVk(vkCreateBuffer(m_device, &bufferInfo, nullptr, &allocation.buffer), "Failed to create buffer");

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(m_device, allocation.buffer, &memoryRequirements);

        VkMemoryAllocateFlagsInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        if (enableDeviceAddress)
        {
            flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        }

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = FindMemoryType(memoryRequirements.memoryTypeBits, properties);
        if (enableDeviceAddress)
        {
            allocateInfo.pNext = &flagsInfo;
        }

        ThrowVk(vkAllocateMemory(m_device, &allocateInfo, nullptr, &allocation.memory), "Failed to allocate buffer memory");
        ThrowVk(vkBindBufferMemory(m_device, allocation.buffer, allocation.memory, 0), "Failed to bind buffer memory");
        return allocation;
    }

    // Free both the VkBuffer and its backing VkDeviceMemory and zero out
    // the allocation so it can be safely re-used or destroyed twice.
    void DestroyBuffer(BufferAllocation& allocation)
    {
        if (allocation.buffer != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(m_device, allocation.buffer, nullptr);
            allocation.buffer = VK_NULL_HANDLE;
        }
        if (allocation.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(m_device, allocation.memory, nullptr);
            allocation.memory = VK_NULL_HANDLE;
        }
        allocation.size = 0;
    }

    VkCommandBuffer BeginSingleTimeCommands() const
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        ThrowVk(vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer),
                "Failed to allocate one-time command buffer");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        ThrowVk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "Failed to begin one-time command buffer");
        return commandBuffer;
    }

    // Submit a one-shot command buffer to the graphics queue, wait for
    // it to finish, and free it. Used for setup work (transfers, BLAS
    // builds) where stalling is acceptable.
    void EndSingleTimeCommands(VkCommandBuffer commandBuffer) const
    {
        ThrowVk(vkEndCommandBuffer(commandBuffer), "Failed to end one-time command buffer");

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        ThrowVk(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE),
                "Failed to submit one-time command buffer");
        ThrowVk(vkQueueWaitIdle(m_graphicsQueue), "Failed to wait for one-time command submission");
        vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
    }

    // Stage host data into a device-local buffer via a temporary
    // host-visible staging buffer + a copy on the graphics queue.
    void UploadToBuffer(const BufferAllocation& allocation, const void* data, size_t dataSize) const
    {
        if (dataSize > static_cast<size_t>(allocation.size))
        {
            throw std::runtime_error("Upload exceeds destination buffer size.");
        }
        void* mapped = nullptr;
        ThrowVk(vkMapMemory(m_device, allocation.memory, 0, allocation.size, 0, &mapped), "Failed to map buffer memory");
        std::memcpy(mapped, data, dataSize);
        vkUnmapMemory(m_device, allocation.memory);
    }

    // Allocate the scene UBO + Mie SSBO and upload the sky parameters. There
    // is no geometry / acceleration structure in this sky-only renderer.
    void CreateSceneResources()
    {
        if (m_commandPool == VK_NULL_HANDLE)
        {
            throw std::runtime_error("Command pool must be created before scene setup.");
        }

        CreateSceneBuffers();
        UploadSceneDataFromConfig();
    }

    VkSurfaceFormatKHR ChooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const
    {
        for (const auto& format : formats)
        {
            if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return format;
            }
        }

        return formats.front();
    }

    VkPresentModeKHR ChoosePresentMode(const std::vector<VkPresentModeKHR>& modes) const
    {
        for (VkPresentModeKHR mode : modes)
        {
            if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
            {
                return mode;
            }
        }

        for (VkPresentModeKHR mode : modes)
        {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                return mode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    // Create (or recreate after a resize) the swapchain and its image
    // views. The swapchain images are created with VK_IMAGE_USAGE_STORAGE_BIT
    // and written directly by the compute shader (no separate offscreen image
    // or blit), so the surface format must support storage usage.
    void CreateSwapchain()
    {
        const auto support = QuerySwapchainSupport(m_physicalDevice);
        const auto surfaceFormat = ChooseSurfaceFormat(support.formats);
        const auto presentMode = ChoosePresentMode(support.presentModes);

        m_swapchainExtent = {m_config.width, m_config.height};
        uint32_t imageCount = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0 && m_config.frameCount > support.capabilities.maxImageCount)
        {
            throw std::runtime_error("Configured frameCount exceeds the swapchain image limit for this surface.");
        }
        if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount)
        {
            imageCount = support.capabilities.maxImageCount;
        }
        imageCount = std::max(imageCount, m_config.frameCount);

        const uint32_t queueFamilyIndices[] = {
            m_queueFamilies.graphicsFamily.value(),
            m_queueFamilies.presentFamily.value(),
        };

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = m_surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = m_swapchainExtent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT;
        createInfo.preTransform = support.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        if (m_queueFamilies.graphicsFamily != m_queueFamilies.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        ThrowVk(vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain),
                "Failed to create Vulkan swapchain");

        m_swapchainFormat = surfaceFormat.format;

        uint32_t actualImageCount = 0;
        ThrowVk(vkGetSwapchainImagesKHR(m_device, m_swapchain, &actualImageCount, nullptr),
                "Failed to query swapchain image count");
        m_swapchainImages.resize(actualImageCount);
        ThrowVk(vkGetSwapchainImagesKHR(m_device, m_swapchain, &actualImageCount, m_swapchainImages.data()),
                "Failed to get swapchain images");
        m_swapchainImageViews.resize(actualImageCount);
        m_swapchainLayouts.assign(actualImageCount, VK_IMAGE_LAYOUT_UNDEFINED);

        for (size_t i = 0; i < m_swapchainImages.size(); ++i)
        {
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = m_swapchainImages[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = m_swapchainFormat;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.layerCount = 1;
            ThrowVk(vkCreateImageView(m_device, &viewInfo, nullptr, &m_swapchainImageViews[i]),
                    "Failed to create swapchain image view");
        }
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
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding sceneBinding{};
        sceneBinding.binding = 2;
        sceneBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sceneBinding.descriptorCount = 1;
        sceneBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding mieBinding{};
        mieBinding.binding = 7;
        mieBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        mieBinding.descriptorCount = 1;
        mieBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        const std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
            binding,
            sceneBinding,
            mieBinding,
        };

        VkDescriptorSetLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        createInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        createInfo.pBindings = bindings.data();
        ThrowVk(vkCreateDescriptorSetLayout(m_device, &createInfo, nullptr, &m_descriptorSetLayout),
                "Failed to create descriptor set layout");
    }

    VkShaderModule CreateShaderModule(const std::vector<char>& bytecode)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = bytecode.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(bytecode.data());

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        ThrowVk(vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule),
                "Failed to create shader module");
        return shaderModule;
    }

    // Compile the compute shader module from its SPIR-V blob and assemble the
    // VkPipeline. Sky-only renderer: a single compute shader evaluates the sky
    // analytically per pixel and writes the swapchain storage image directly.
    void CreatePipeline()
    {
        const auto computeBytecode = LoadBinaryFile(L"path_tracer.comp.spv");
        VkShaderModule computeModule = CreateShaderModule(computeBytecode);

        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = computeModule;
        stageInfo.pName = "main";

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_descriptorSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;
        ThrowVk(vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_pipelineLayout),
                "Failed to create pipeline layout");

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = m_pipelineLayout;
        ThrowVk(vkCreateComputePipelines(m_device,
                                         VK_NULL_HANDLE,
                                         1,
                                         &pipelineInfo,
                                         nullptr,
                                         &m_computePipeline),
                "Failed to create compute pipeline");

        vkDestroyShaderModule(m_device, computeModule, nullptr);
    }

    // Allocate the descriptor pool and one descriptor set per layout
    // slot. Initial bindings are deferred to UpdateDescriptorSetContents()
    // because they depend on scene resources that are created later.
    void CreateDescriptorSets()
    {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSize.descriptorCount = static_cast<uint32_t>(m_swapchainImageViews.size());
        VkDescriptorPoolSize scenePoolSize{};
        scenePoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        scenePoolSize.descriptorCount = static_cast<uint32_t>(m_swapchainImageViews.size());
        VkDescriptorPoolSize storageBufferPoolSize{};
        storageBufferPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        storageBufferPoolSize.descriptorCount = static_cast<uint32_t>(m_swapchainImageViews.size());
        const std::array<VkDescriptorPoolSize, 3> poolSizes = {
            poolSize,
            scenePoolSize,
            storageBufferPoolSize,
        };

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = static_cast<uint32_t>(m_swapchainImageViews.size());
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        ThrowVk(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool),
                "Failed to create descriptor pool");

        std::vector<VkDescriptorSetLayout> layouts(m_swapchainImageViews.size(), m_descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        allocInfo.pSetLayouts = layouts.data();

        m_descriptorSets.resize(layouts.size());
        ThrowVk(vkAllocateDescriptorSets(m_device, &allocInfo, m_descriptorSets.data()),
                "Failed to allocate descriptor sets");
        UpdateDescriptorSetContents();
    }

    // Create the graphics-queue command pool used for both per-frame
    // command buffers and one-shot setup commands.
    void CreateCommandPool()
    {
        VkCommandPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        createInfo.queueFamilyIndex = m_queueFamilies.graphicsFamily.value();
        ThrowVk(vkCreateCommandPool(m_device, &createInfo, nullptr, &m_commandPool),
                "Failed to create command pool");
    }

    // Allocate one primary command buffer per frame in flight.
    void CreateCommandBuffers()
    {
        m_commandBuffers.resize(m_config.frameCount);
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(m_commandBuffers.size());
        ThrowVk(vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data()),
                "Failed to allocate command buffers");
    }

    // Allocate the per-frame semaphores (image-available, render-
    // finished) and fences that gate command-buffer reuse.
    void CreateSyncObjects()
    {
        m_frames.resize(m_config.frameCount);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (auto& frame : m_frames)
        {
            ThrowVk(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &frame.imageAvailable),
                    "Failed to create image-available semaphore");
            ThrowVk(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &frame.renderFinished),
                    "Failed to create render-finished semaphore");
            ThrowVk(vkCreateFence(m_device, &fenceInfo, nullptr, &frame.inFlight),
                    "Failed to create frame fence");
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
    void RecordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        ThrowVk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "Failed to begin command buffer");

        VkImageMemoryBarrier toGeneral{};
        toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toGeneral.oldLayout = m_swapchainLayouts[imageIndex];
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.image = m_swapchainImages[imageIndex];
        toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toGeneral.subresourceRange.levelCount = 1;
        toGeneral.subresourceRange.layerCount = 1;
        toGeneral.srcAccessMask = 0;
        toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &toGeneral);

        const PushConstants pushConstants = BuildPushConstants();
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_pipelineLayout,
                                0,
                                1,
                                &m_descriptorSets[imageIndex],
                                0,
                                nullptr);
        vkCmdPushConstants(commandBuffer,
                           m_pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(PushConstants),
                           &pushConstants);
        // 8x8 workgroups, matching the compute shader's local size; round up so
        // the whole render target is covered (the shader discards the overhang).
        constexpr uint32_t kTile = 8;
        const uint32_t groupsX = (m_swapchainExtent.width + kTile - 1) / kTile;
        const uint32_t groupsY = (m_swapchainExtent.height + kTile - 1) / kTile;
        vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);

        VkImageMemoryBarrier toPresent = toGeneral;
        toPresent.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        toPresent.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        toPresent.dstAccessMask = 0;

        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &toPresent);

        ThrowVk(vkEndCommandBuffer(commandBuffer), "Failed to end command buffer");
        m_swapchainLayouts[imageIndex] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }

    // Wait for the current in-flight slot, acquire a swapchain image,
    // record + submit the frame's command buffer, then present.
    // Handles VK_ERROR_OUT_OF_DATE_KHR by recreating the swapchain.
    void RenderFrame()
    {
        FrameResources& frame = m_frames[m_currentFrame];
        ThrowVk(vkWaitForFences(m_device, 1, &frame.inFlight, VK_TRUE, UINT64_MAX), "Failed to wait for fence");

        uint32_t imageIndex = 0;
        const VkResult acquire = vkAcquireNextImageKHR(m_device,
                                                       m_swapchain,
                                                       UINT64_MAX,
                                                       frame.imageAvailable,
                                                       VK_NULL_HANDLE,
                                                       &imageIndex);
        if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR)
        {
            ThrowVk(acquire, "Failed to acquire swapchain image");
        }

        ThrowVk(vkResetFences(m_device, 1, &frame.inFlight), "Failed to reset fence");
        ThrowVk(vkResetCommandBuffer(m_commandBuffers[m_currentFrame], 0), "Failed to reset command buffer");
        RecordCommandBuffer(m_commandBuffers[m_currentFrame], imageIndex);

        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &frame.imageAvailable;
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_commandBuffers[m_currentFrame];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &frame.renderFinished;
        ThrowVk(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, frame.inFlight), "Failed to submit command buffer");

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &frame.renderFinished;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &m_swapchain;
        presentInfo.pImageIndices = &imageIndex;
        const VkResult present = vkQueuePresentKHR(m_presentQueue, &presentInfo);
        if (present != VK_SUCCESS && present != VK_SUBOPTIMAL_KHR)
        {
            ThrowVk(present, "Failed to present swapchain image");
        }

        ++m_frameIndex;
        m_currentFrame = (m_currentFrame + 1) % static_cast<uint32_t>(m_frames.size());
    }

    // Throttled window-title update displaying live FPS and per-frame
    // milliseconds — cheap diagnostic for performance tuning.
    void UpdateWindowTitle(double fps, double frameMs)
    {
        wchar_t buffer[256]{};
        std::swprintf(buffer,
                      sizeof(buffer) / sizeof(buffer[0]),
                      L"Vulkan Path Tracer - %.1f FPS (%.2f ms)",
                      fps,
                      frameMs);
        SetWindowTextW(m_window, buffer);
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

    // Idle the device, then destroy every Vulkan handle in reverse
    // construction order. Idempotent — safe to invoke from the
    // destructor after a successful run or after a partial-init throw.
    void Cleanup()
    {
        if (m_device != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(m_device);
        }

        for (auto& frame : m_frames)
        {
            if (frame.inFlight != VK_NULL_HANDLE) vkDestroyFence(m_device, frame.inFlight, nullptr);
            if (frame.renderFinished != VK_NULL_HANDLE) vkDestroySemaphore(m_device, frame.renderFinished, nullptr);
            if (frame.imageAvailable != VK_NULL_HANDLE) vkDestroySemaphore(m_device, frame.imageAvailable, nullptr);
        }

        if (!m_commandBuffers.empty() && m_commandPool != VK_NULL_HANDLE)
        {
            vkFreeCommandBuffers(m_device,
                                 m_commandPool,
                                 static_cast<uint32_t>(m_commandBuffers.size()),
                                 m_commandBuffers.data());
        }
        if (m_commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        if (m_descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        if (m_computePipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_device, m_computePipeline, nullptr);
        if (m_pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        if (m_descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
        DestroySceneBuffers();
        for (VkImageView view : m_swapchainImageViews)
        {
            if (view != VK_NULL_HANDLE) vkDestroyImageView(m_device, view, nullptr);
        }
        if (m_swapchain != VK_NULL_HANDLE) vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        if (m_device != VK_NULL_HANDLE) vkDestroyDevice(m_device, nullptr);
        if (m_surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        if (m_instance != VK_NULL_HANDLE) vkDestroyInstance(m_instance, nullptr);
        if (m_window != nullptr) DestroyWindow(m_window);
    }

    struct FrameResources
    {
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
    };

    HWND m_window = nullptr;

    VkInstance m_instance = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VkQueue m_presentQueue = VK_NULL_HANDLE;
    QueueFamilyIndices m_queueFamilies;

    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkFormat m_swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D m_swapchainExtent{};
    std::vector<VkImage> m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;
    std::vector<VkImageLayout> m_swapchainLayouts;

    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_computePipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_descriptorSets;

    BufferAllocation m_sceneDataBuffer{};
    BufferAllocation m_mieScatteringBuffer{};

    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_commandBuffers;
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
    // major axis (measured in the camera image plane from the right axis); in
    // elliptical mode [ and ] adjust ellipticity (-45..+45 degrees).
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
