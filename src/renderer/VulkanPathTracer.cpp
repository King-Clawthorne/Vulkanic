// VulkanPathTracer.cpp — the renderer: GLFW window, Vulkan setup, and the frame loop.
//
// A polarized-sky simulator with no scene geometry. Each frame one compute pass
// accumulates sky radiance into an HDR buffer, a second models the camera and
// display into the swapchain image, and Dear ImGui draws the control panel on top.

#include "VulkanPathTracer.h"

#include "app/CameraController.h"
#include "config/RuntimeConfig.h"
#include "sky/MieScattering.h"
#include "sky/RainbowScattering.h"

#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <VkBootstrap.h>
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <imgui_impl_glfw.h>
// VMA-Hpp targets a newer vulkan-hpp than SDK 1.4.321, which lacks this macro.
#ifndef VULKAN_HPP_DISPATCH_LOADER_STATIC_TYPE
#define VULKAN_HPP_DISPATCH_LOADER_STATIC_TYPE VULKAN_HPP_NAMESPACE::detail::DispatchLoaderStatic
#endif
#include <vk_mem_alloc.hpp>
#include <spirv_reflect.h>

#include <array>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <print>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

constexpr std::array kShaderFiles{L"path_tracer.comp.spv", L"post_process.comp.spv"};

static std::string LoadSpirv(const wchar_t* fileName)
{
    const auto path = ResolveRuntimeFilePath(fileName);
    if (path.empty()) throw std::runtime_error("Failed to find SPIR-V shader.");
    return LoadTextFile(path);
}

// A VMA buffer and its memory. Members are destroyed in reverse order, so the
// buffer is released before its allocation.
struct GpuBuffer {
    vma::UniqueAllocation allocation;
    vma::UniqueBuffer buffer;
};

// Sky uniforms. Layout mirrors the shader's std140 block; the static_assert
// catches accidental reshuffles.
struct alignas(16) SceneData {
    // x = Rayleigh extinction at 550 nm, y = Mie extinction,
    // z = solar temperature (K), w = solar radiance at 550 nm.
    float skySpectralParams[4];
    float skyRadiiScaleHeights[4];
    float skySunDirectionRadius[4];
    uint32_t skySampleCounts[4];
    // x = sun-disk AA width, y = Rayleigh depolarization,
    // z = Mie table angle bins, w = Lambertian ground albedo.
    float skyVrtParams[4];
    float rainbowCenterEnabled[4];
    float rainbowRadiiEdge[4];
    float rainbowOptical[4];
    uint32_t rainbowMultiple[4];
    float spectralBands[kSpectralBandCount][4]; // Rayleigh extinction, solar radiance
};

static_assert(sizeof(SceneData) == 352, "Scene data layout must stay 16-byte aligned.");

struct PushConstants {
    float cameraPositionFrame[4];
    float cameraForwardSamples[4];
    float cameraRightBounces[4];
    float cameraUpTanHalfFovY[4];
    // x = exposure, y = viewport aspect ratio, z/w unused.
    float displayParams[4];
    // x/y/z = polarization analyzer, w unused.
    float polarizer[4];
    uint32_t imageSize[2];
};

static_assert(sizeof(PushConstants) == 104, "Push constant layout must match the shader.");

// Owns the window and every Vulkan handle, and runs the frame loop.
class VulkanPathTracer {
public:
    // Members release themselves in reverse declaration order once the GPU
    // is idle; only ImGui and GLFW need explicit shutdown.
    ~VulkanPathTracer()
    {
        if (*m_device) {
            m_device.waitIdle();
        }
        if (m_imguiInitialized) {
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            m_imguiInitialized = false;
        }
        if (m_window != nullptr) {
            glfwDestroyWindow(m_window);
            m_window = nullptr;
        }
        glfwTerminate();
    }

    void Run()
    {
        ApplyRuntimeConfig(m_configFile.Load(), true);
        CreateWindowAndShow();
        const vkb::Instance vkbInstance = CreateInstance();
        CreateSurface();
        CreateLogicalDevice(PickPhysicalDevice(vkbInstance));
        CreateAllocator();
        CreateCommandPool();
        CreateSceneResources();
        CreateSwapchain();
        CreateDescriptorSetLayout();
        CreatePipeline();
        CreateFrameResources();
        CreateGui();
        MessageLoop();
        m_device.waitIdle();
    }

private:
    // F1 toggles the GUI; F2 cycles config files; F5 saves the current one.
    // Installed before ImGui's GLFW backend, which chains to it.
    static void KeyCallback(GLFWwindow* window, int key, int, int action, int)
    {
        auto* app = static_cast<VulkanPathTracer*>(glfwGetWindowUserPointer(window));
        if (app == nullptr || action != GLFW_PRESS) {
            return;
        }
        if (key == GLFW_KEY_F1) app->m_showGui = !app->m_showGui;
        if (key == GLFW_KEY_F2) app->m_cycleConfigRequested = true;
        if (key == GLFW_KEY_F5) app->m_saveConfigRequested = true;
    }

    void CreateGui()
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForVulkan(m_window, true);

        ImGui_ImplVulkan_InitInfo initInfo{};
        initInfo.ApiVersion = VK_API_VERSION_1_4;
        initInfo.Instance = static_cast<VkInstance>(*m_instance);
        initInfo.PhysicalDevice = static_cast<VkPhysicalDevice>(*m_physicalDevice);
        initInfo.Device = static_cast<VkDevice>(*m_device);
        initInfo.QueueFamily = m_queueFamily;
        initInfo.Queue = static_cast<VkQueue>(*m_graphicsQueue);
        initInfo.DescriptorPoolSize = 64;
        initInfo.MinImageCount = std::max(2u, m_config.render.frameCount);
        initInfo.ImageCount = static_cast<uint32_t>(m_swapchainImages.size());
        initInfo.UseDynamicRendering = true;
        initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
        initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats =
            reinterpret_cast<const VkFormat*>(&m_swapchainFormat);
        initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        if (!ImGui_ImplVulkan_Init(&initInfo)) {
            throw std::runtime_error("Failed to initialize Dear ImGui Vulkan backend.");
        }
        m_imguiInitialized = true;
    }

    void BuildGuiFrame()
    {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (m_showGui) {
            ImGui::SetNextWindowPos({16.0f, 16.0f}, ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize({330.0f, 0.0f}, ImGuiCond_FirstUseEver);
            ImGui::Begin("Vulkanic Controls", &m_showGui, ImGuiWindowFlags_AlwaysAutoResize);
            RuntimeConfig next = m_config;
            bool changed = false;
            // Each control edits a field of `next`; `scale` converts the stored
            // value into friendlier display units (e.g. 1e-6 m^-1 -> 1).
            const auto slider = [&](const char* label, float& value, float min, float max,
                                    float scale = 1.0f, const char* format = "%.2f") {
                float shown = value * scale;
                if (!ImGui::SliderFloat(label, &shown, min, max, format)) return false;
                value = shown / scale;
                return changed = true;
            };
            const auto sliderInt = [&](const char* label, uint32_t& value, int min, int max) {
                int shown = static_cast<int>(value);
                if (!ImGui::SliderInt(label, &shown, min, max)) return false;
                value = static_cast<uint32_t>(shown);
                return changed = true;
            };

            SkySpectralConfig& sky = next.sky.spectral;
            slider("Exposure", next.sky.exposure, 0.1f, 4.0f);
            slider("Aerosol density", sky.betaMie, 0.0f, 100.0f, 1.0e6f, "%.1f");
            sliderInt("View steps", sky.viewSteps, 1, 20);
            sliderInt("Scattering orders", sky.scatteringOrders, 1, 3);
            slider("Ground albedo", sky.groundAlbedo, 0.0f, 1.0f);

            bool rainbowEnabled = next.rainbow.enabled != 0;
            if (ImGui::Checkbox("Rainbow", &rainbowEnabled)) {
                next.rainbow.enabled = rainbowEnabled ? 1u : 0u;
                changed = true;
            }
            float rainScattering = next.rainbow.scatteringCoefficient;
            if (rainbowEnabled && slider("Rain scattering", rainScattering, 0.0f, 5.0f, 1.0e4f)) {
                SetRainbowScattering(next.rainbow, rainScattering);
            }
            if (changed) {
                ApplyRuntimeConfig(next, false);
            }
            ImGui::Separator();
            ImGui::Text("Config: %s", m_configFile.Path().filename().string().c_str());
            ImGui::TextDisabled("F1 GUI | F2 next config | F5 save");
            ImGui::End();
        }
        ImGui::Render();
    }

    SceneData BuildSceneData() const
    {
        const SkySpectralConfig& s = m_config.sky.spectral;
        const RainbowConfig& r = m_config.rainbow;
        SceneData sceneData{
            .skySpectralParams = {s.betaRayleigh550, s.betaMie, s.sunTemperatureKelvin, s.sunRadiance550},
            .skyRadiiScaleHeights = {s.earthRadius, s.atmosphereRadius, s.scaleHeightRayleigh, s.scaleHeightMie},
            .skySunDirectionRadius = {s.sunDirection[0], s.sunDirection[1], s.sunDirection[2], s.sunRadius},
            .skySampleCounts = {s.secondarySamples, s.viewSteps, s.samples, s.scatteringOrders},
            .skyVrtParams = {s.sunAa, s.rayleighDepolarization, float(s.mieTableAngleBins), s.groundAlbedo},
            .rainbowCenterEnabled = {r.center.x, r.center.y, r.center.z, float(r.enabled)},
            .rainbowRadiiEdge = {r.radii.x, r.radii.y, r.radii.z, r.edgeSoftness},
            .rainbowOptical = {r.scatteringCoefficient, r.extinctionCoefficient, float(r.angleBins), float(r.viewSteps)},
            .rainbowMultiple = {r.scatteringOrders, r.multipleScatteringSamples, r.multipleScatteringSteps, 0},
        };
        // Cache the existing per-wavelength formulas on config upload. They
        // depend on scene parameters, not the pixel, path or scattering order.
        // Planck radiance remains normalized at 550 nm, with the same float
        // wavelength grid and constants previously evaluated in sky.comp.
        for (int band = 0; band < kSpectralBandCount; ++band) {
            const float wavelength = float(kSpectralLambdaMinNm + kSpectralLambdaStepNm * band);
            const float ratio = 550.0f / wavelength;
            const float lambda = wavelength * 1.0e-9f;
            constexpr float reference = 550.0e-9f;
            constexpr float c2 = 1.4387769e-2f;
            const float shape = std::pow(reference / lambda, 5.0f) * (std::exp(c2 / (reference * s.sunTemperatureKelvin)) - 1.0f) / (std::exp(c2 / (lambda * s.sunTemperatureKelvin)) - 1.0f);
            sceneData.spectralBands[band][0] = s.betaRayleigh550 * ratio * ratio * ratio * ratio;
            sceneData.spectralBands[band][1] = s.sunRadiance550 * shape;
        }
        return sceneData;
    }

    void CreateSceneBuffers()
    {
        m_sceneDataBuffer = CreateBuffer(sizeof(SceneData), vk::BufferUsageFlagBits::eUniformBuffer);
        const VkDeviceSize accumulationSize = static_cast<VkDeviceSize>(m_config.render.width) * static_cast<VkDeviceSize>(m_config.render.height) * sizeof(float) * 4u;
        m_accumulationBuffer = CreateBuffer(accumulationSize, vk::BufferUsageFlagBits::eStorageBuffer);
        CreateMieScatteringBuffer();
        CreateRainbowScatteringBuffer();
    }

    MieAerosolParams BuildMieAerosolParams() const
    {
        const SkySpectralConfig& s = m_config.sky.spectral;
        MieAerosolParams params{};
        params.refractiveIndexReal = s.aerosolRefractiveIndexReal;
        params.refractiveIndexImag = s.aerosolRefractiveIndexImag;
        params.meanRadiusMicrometers = s.aerosolMeanRadiusMicrometers;
        params.sigma = s.aerosolSigma;
        params.angleBins = static_cast<int>(s.mieTableAngleBins);
        return params;
    }

    // Bake the Lorenz–Mie scattering matrix on the CPU (binding 7).
    void CreateMieScatteringBuffer()
    {
        const MieAerosolParams params = BuildMieAerosolParams();
        const std::vector<MieMatrixEntry> table = ComputeMieScatteringTable(params);
        const VkDeviceSize size = static_cast<VkDeviceSize>(table.size() * sizeof(MieMatrixEntry));
        m_mieScatteringBuffer = CreateBuffer(size, vk::BufferUsageFlagBits::eStorageBuffer);
        UploadToBuffer(m_mieScatteringBuffer, std::as_bytes(std::span{table}));
        std::println("[Sky] Baked Lorenz-Mie scattering matrix: {} angle bins x {} spectral bands.",
                     params.angleBins, kSpectralBandCount);
    }

    RainbowScatteringParams BuildRainbowScatteringParams() const
    {
        RainbowScatteringParams params{};
        params.effectiveRadiusMicrometers = m_config.rainbow.effectiveRadiusMicrometers;
        params.effectiveVariance = m_config.rainbow.effectiveVariance;
        params.solarAngularRadiusRadians = m_config.sky.spectral.sunRadius;
        params.angleBins = static_cast<int>(m_config.rainbow.angleBins);
        params.includeSecondary = m_config.rainbow.includeSecondary != 0;
        return params;
    }

    void CreateRainbowScatteringBuffer()
    {
        const RainbowScatteringParams params = BuildRainbowScatteringParams();
        std::vector<MieMatrixEntry> table = ComputeRainbowScatteringTable(params);
        AppendRainbowSamplingCdf(table, params.angleBins);
        const VkDeviceSize size = static_cast<VkDeviceSize>(table.size() * sizeof(MieMatrixEntry));
        m_rainbowScatteringBuffer = CreateBuffer(size, vk::BufferUsageFlagBits::eStorageBuffer);
        UploadToBuffer(m_rainbowScatteringBuffer, std::as_bytes(std::span{table}));
        std::println("[Rainbow] Baked droplet Mueller matrix: {} angle bins x {} spectral bands.",
                     params.angleBins, kSpectralBandCount);
    }

    void UploadSceneDataFromConfig()
    {
        const SceneData sceneData = BuildSceneData();
        UploadToBuffer(m_sceneDataBuffer, std::as_bytes(std::span{&sceneData, 1}));
    }

    // Re-upload sky parameters, rebuilding the Mie / rainbow tables if asked.
    void RefreshSceneFromConfig(bool rebuildMieTable, bool rebuildRainbowTable)
    {
        if (!m_sceneDataBuffer.buffer) {
            return;
        }

        m_device.waitIdle();
        if (rebuildMieTable) {
            CreateMieScatteringBuffer();
        }
        if (rebuildRainbowTable) {
            CreateRainbowScatteringBuffer();
        }
        UploadSceneDataFromConfig();
    }

    // Apply a new config, doing only the refresh work its changes require.
    void ApplyRuntimeConfig(const RuntimeConfig& config, bool resetCameraState)
    {
        const bool pipelineChanged = config.sky.spectral.scatteringOrders != m_config.sky.spectral.scatteringOrders || config.sky.spectral.viewSteps != m_config.sky.spectral.viewSteps || config.sky.spectral.samples != m_config.sky.spectral.samples || config.sky.spectral.secondarySamples != m_config.sky.spectral.secondarySamples;
        const bool skySpectralChanged = config.sky.spectral != m_config.sky.spectral;
        const bool mieAerosolChanged = HasMieAerosolChanged(config.sky.spectral, m_config.sky.spectral);
        const bool rainbowChanged = config.rainbow != m_config.rainbow;
        const bool rainbowOpticsChanged = HasRainbowOpticsChanged(config.rainbow, m_config.rainbow) || config.sky.spectral.sunRadius != m_config.sky.spectral.sunRadius;
        if (skySpectralChanged || rainbowChanged || config.render.samplesPerPixel != m_config.render.samplesPerPixel || config.camera.fovYDegrees != m_config.camera.fovYDegrees)
            m_accumulationResetRequested = true;

        if (!resetCameraState) {
            if (config.render.width != m_config.render.width || config.render.height != m_config.render.height || config.render.frameCount != m_config.render.frameCount || config.render.vsync != m_config.render.vsync) {
                std::println("[Config] width/height/frameCount/vsync changes apply on the next launch.");
            }
        }

        m_config = config;
        if (resetCameraState) {
            m_camera.Reset(m_config);
            return;
        }

        m_camera.ClampPitch(m_config);

        if (skySpectralChanged || rainbowChanged) {
            RefreshSceneFromConfig(mieAerosolChanged, rainbowOpticsChanged);
            if (pipelineChanged) CreatePipeline();
        }
    }

    // Called once per frame: apply F2 (cycle) / F5 (save) requests and pick
    // up edits made to the active config file on disk.
    void ProcessConfigChanges()
    {
        if (std::exchange(m_cycleConfigRequested, false)) {
            if (auto config = m_configFile.CycleNext()) ApplyRuntimeConfig(*config, false);
        }
        if (std::exchange(m_saveConfigRequested, false)) {
            m_configFile.Save(m_config);
        }
        if (auto config = m_configFile.ReloadIfChanged()) ApplyRuntimeConfig(*config, false);
    }

    void CreateWindowAndShow()
    {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW.");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        m_window = glfwCreateWindow(static_cast<int>(m_config.render.width), static_cast<int>(m_config.render.height),
                                    "Vulkan Path Tracer - FPS: measuring...", nullptr, nullptr);
        if (m_window == nullptr) {
            throw std::runtime_error("Failed to create window.");
        }
        glfwSetWindowUserPointer(m_window, this);
        glfwSetKeyCallback(m_window, KeyCallback);

        std::println("[Config] Edit {} and save to hot-reload tuning.", m_configFile.Path().string());
        std::println("[Config] width, height, frameCount, and vsync are loaded from JSON at startup.");
        std::println("[Controls] Hold RMB or use the arrow keys to look around the sky. R resets the view.");
        std::println("[Controls] P toggles the polarization filter; C switches linear/elliptical.");
        std::println("[Controls] Linear: [ ] rotate the filter axis. Elliptical: [ ] adjust ellipticity.");
        std::println("[Controls] F1 toggles the live GUI control panel.");
    }

    // vk-bootstrap enables the platform surface extensions; validation layers
    // are used only when installed.
    vkb::Instance CreateInstance()
    {
        auto instanceResult = vkb::InstanceBuilder{}
                                  .set_app_name("Vulkan Path Tracer")
                                  .set_engine_name("None")
                                  .require_api_version(1, 4, 0)
                                  .enable_extension(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME)
                                  .enable_extension(VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME)
                                  .request_validation_layers()
                                  .build();
        if (!instanceResult) {
            throw std::runtime_error("Failed to create Vulkan instance: " + instanceResult.error().message());
        }

        m_instance = vk::raii::Instance(m_context, instanceResult.value().instance);
        return instanceResult.value();
    }

    void CreateSurface()
    {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        vk::detail::resultCheck(vk::Result(glfwCreateWindowSurface(static_cast<VkInstance>(*m_instance), m_window, nullptr, &surface)),
                                "Failed to create window surface");
        m_surface = vk::raii::SurfaceKHR(m_instance, surface);
    }

    // The compute shader writes the swapchain image directly, so the surface
    // must support storage usage. Non-semantic info keeps debugPrintfEXT working.
    vkb::PhysicalDevice PickPhysicalDevice(const vkb::Instance& vkbInstance)
    {
        VkPhysicalDeviceFeatures requiredFeatures{};
        requiredFeatures.shaderStorageImageWriteWithoutFormat = VK_TRUE;
        VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        features13.dynamicRendering = VK_TRUE;
        features13.synchronization2 = VK_TRUE;
        VkPhysicalDeviceVulkan14Features features14{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES};
        features14.pushDescriptor = VK_TRUE;
        VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR swapchainMaintenance{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR};
        swapchainMaintenance.swapchainMaintenance1 = VK_TRUE;

        auto deviceResult = vkb::PhysicalDeviceSelector{vkbInstance}
                                .set_surface(static_cast<VkSurfaceKHR>(*m_surface))
                                .set_minimum_version(1, 4)
                                .set_required_features(requiredFeatures)
                                .set_required_features_13(features13)
                                .add_required_extension_features(features14)
                                .add_required_extension_features(swapchainMaintenance)
                                .add_required_extension(VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME)
                                .add_required_extension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME)
                                .require_present()
                                .select();
        if (!deviceResult) {
            throw std::runtime_error("Failed to select Vulkan physical device: " + deviceResult.error().message());
        }

        m_physicalDevice = vk::raii::PhysicalDevice(m_instance, deviceResult.value().physical_device);

        const vk::SurfaceCapabilitiesKHR capabilities = m_physicalDevice.getSurfaceCapabilitiesKHR(*m_surface);
        if (!(capabilities.supportedUsageFlags & vk::ImageUsageFlagBits::eStorage)) {
            throw std::runtime_error("Selected GPU does not support storage-image swapchains for this app.");
        }
        return deviceResult.value();
    }

    void CreateLogicalDevice(const vkb::PhysicalDevice& vkbPhysicalDevice)
    {
        auto deviceResult = vkb::DeviceBuilder{vkbPhysicalDevice}.build();
        if (!deviceResult) {
            throw std::runtime_error("Failed to create Vulkan logical device: " + deviceResult.error().message());
        }
        m_device = vk::raii::Device(m_physicalDevice, deviceResult.value().device);

        const auto queueFamily = deviceResult.value().get_queue_index(vkb::QueueType::graphics);
        if (!queueFamily || !m_physicalDevice.getSurfaceSupportKHR(queueFamily.value(), *m_surface)) {
            throw std::runtime_error("The graphics queue cannot present to the window.");
        }
        m_queueFamily = queueFamily.value();
        m_graphicsQueue = m_device.getQueue(m_queueFamily, 0);
    }

    void CreateAllocator()
    {
        m_allocator = vma::createAllocatorUnique(vma::AllocatorCreateInfo{}
                                                     .setInstance(*m_instance)
                                                     .setPhysicalDevice(*m_physicalDevice)
                                                     .setDevice(*m_device)
                                                     .setVulkanApiVersion(VK_API_VERSION_1_4));
    }

    // Every buffer here is small and CPU-written, so all are host-visible.
    GpuBuffer CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage) const
    {
        auto [allocation, buffer] = m_allocator->createBufferUnique(
            vk::BufferCreateInfo{{}, size, usage},
            vma::AllocationCreateInfo{vma::AllocationCreateFlagBits::eHostAccessSequentialWrite, vma::MemoryUsage::eAuto});
        return {std::move(allocation), std::move(buffer)};
    }

    void UploadToBuffer(const GpuBuffer& buffer, std::span<const std::byte> data) const
    {
        m_allocator->copyMemoryToAllocation(data.data(), *buffer.allocation, 0, data.size());
    }

    void CreateSceneResources()
    {
        CreateSceneBuffers();
        UploadSceneDataFromConfig();
    }

    // Present mode is FIFO with vsync, otherwise IMMEDIATE, falling back to MAILBOX.
    void CreateSwapchain()
    {
        vkb::SwapchainBuilder builder{*m_physicalDevice, *m_device, *m_surface, m_queueFamily, m_queueFamily};
        builder.set_desired_format({VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
            .set_desired_present_mode(m_config.render.vsync ? VK_PRESENT_MODE_FIFO_KHR : VK_PRESENT_MODE_IMMEDIATE_KHR)
            .add_fallback_present_mode(m_config.render.vsync ? VK_PRESENT_MODE_FIFO_KHR : VK_PRESENT_MODE_MAILBOX_KHR)
            .set_desired_extent(m_config.render.width, m_config.render.height)
            .set_image_usage_flags(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
            .set_required_min_image_count(m_config.render.frameCount);
        auto swapchainResult = builder.build();
        if (!swapchainResult) {
            throw std::runtime_error("Failed to create Vulkan swapchain: " + swapchainResult.error().message());
        }

        vkb::Swapchain swapchain = swapchainResult.value();
        m_swapchain = vk::raii::SwapchainKHR(m_device, swapchain.swapchain);
        m_swapchainFormat = static_cast<vk::Format>(swapchain.image_format);
        m_swapchainExtent = swapchain.extent;

        m_swapchainImages = m_swapchain.getImages();
        m_swapchainImageViews.clear();
        for (VkImageView view : swapchain.get_image_views().value()) {
            m_swapchainImageViews.emplace_back(m_device, view);
        }
    }

    // Read the descriptor bindings and push-constant block from both compute
    // shaders, so the layout can never drift from the GLSL.
    void CreateDescriptorSetLayout()
    {
        std::map<uint32_t, vk::DescriptorSetLayoutBinding> bindings;
        for (const wchar_t* fileName : kShaderFiles) {
            const std::string spirv = LoadSpirv(fileName);
            const spv_reflect::ShaderModule module(spirv.size(), spirv.data());
            if (module.GetResult() != SPV_REFLECT_RESULT_SUCCESS) throw std::runtime_error("Failed to reflect SPIR-V shader.");

            uint32_t count = 0;
            module.EnumerateDescriptorBindings(&count, nullptr);
            std::vector<SpvReflectDescriptorBinding*> reflected(count);
            module.EnumerateDescriptorBindings(&count, reflected.data());
            for (const SpvReflectDescriptorBinding* binding : reflected) {
                bindings[binding->binding] = {binding->binding, vk::DescriptorType(binding->descriptor_type), binding->count,
                                              vk::ShaderStageFlagBits::eCompute};
            }

            module.EnumeratePushConstantBlocks(&count, nullptr);
            std::vector<SpvReflectBlockVariable*> blocks(count);
            module.EnumeratePushConstantBlocks(&count, blocks.data());
            for (const SpvReflectBlockVariable* block : blocks) {
                if (block->offset + block->size != sizeof(PushConstants)) {
                    throw std::runtime_error("PushConstants does not match the shader's push-constant block.");
                }
            }
        }

        // Push descriptors: bindings are written straight into the command
        // buffer each frame, so no descriptor pool or sets are needed.
        const auto layoutBindings = bindings | std::views::values | std::ranges::to<std::vector>();
        vk::DescriptorSetLayoutCreateInfo createInfo{vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor};
        createInfo.setBindings(layoutBindings);
        m_descriptorSetLayout = vk::raii::DescriptorSetLayout(m_device, createInfo);
    }

    vk::raii::ShaderModule CreateShaderModule(const wchar_t* fileName)
    {
        const std::string bytecode = LoadSpirv(fileName);
        return vk::raii::ShaderModule(m_device, vk::ShaderModuleCreateInfo{{}, bytecode.size(), std::bit_cast<const uint32_t*>(bytecode.data())});
    }

    void CreatePipeline()
    {
        const vk::raii::ShaderModule computeModule = CreateShaderModule(kShaderFiles[0]);
        const vk::raii::ShaderModule postModule = CreateShaderModule(kShaderFiles[1]);

        // Specialize loop bounds without changing the requested quality. The
        // config refresh waits for in-flight work before rebuilding pipelines.
        const std::array<uint32_t, 4> settings = {m_config.sky.spectral.scatteringOrders,
                                                  m_config.sky.spectral.viewSteps, m_config.sky.spectral.samples,
                                                  m_config.sky.spectral.secondarySamples};
        const std::array<vk::SpecializationMapEntry, 4> entries = {{{0, 0, sizeof(uint32_t)}, {1, 4, sizeof(uint32_t)}, {2, 8, sizeof(uint32_t)}, {3, 12, sizeof(uint32_t)}}};
        const vk::SpecializationInfo specialization{uint32_t(entries.size()), entries.data(), sizeof(settings), settings.data()};

        const vk::PushConstantRange pushRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants)};
        vk::PipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.setSetLayouts(*m_descriptorSetLayout);
        layoutInfo.setPushConstantRanges(pushRange);
        m_pipelineLayout = vk::raii::PipelineLayout(m_device, layoutInfo);

        const auto makePipeline = [&](const vk::raii::ShaderModule& module, const vk::SpecializationInfo* info) {
            const vk::PipelineShaderStageCreateInfo stage{{}, vk::ShaderStageFlagBits::eCompute, *module, "main", info};
            return vk::raii::Pipeline(m_device, nullptr, vk::ComputePipelineCreateInfo{{}, stage, *m_pipelineLayout});
        };
        m_computePipeline = makePipeline(computeModule, &specialization);
        m_postProcessPipeline = makePipeline(postModule, nullptr);
    }

    void CreateCommandPool()
    {
        vk::CommandPoolCreateInfo createInfo{vk::CommandPoolCreateFlagBits::eResetCommandBuffer, m_queueFamily};
        m_commandPool = vk::raii::CommandPool(m_device, createInfo);
    }

    void CreateFrameResources()
    {
        const uint32_t count = m_config.render.frameCount;
        vk::raii::CommandBuffers commandBuffers(m_device, {*m_commandPool, vk::CommandBufferLevel::ePrimary, count});
        const vk::FenceCreateInfo signaled{vk::FenceCreateFlagBits::eSignaled};
        m_frames.clear();
        for (uint32_t i = 0; i < count; ++i) {
            m_frames.push_back({std::move(commandBuffers[i]), m_device.createSemaphore({}), m_device.createSemaphore({}),
                                m_device.createFence(signaled), m_device.createFence(signaled)});
        }
    }

    PushConstants BuildPushConstants()
    {
        const Vec3 position = m_camera.Position();
        const Vec3 forward = m_camera.Forward();
        const Vec3 right = Normalize(Cross({0.0f, 1.0f, 0.0f}, forward));
        const Vec3 up = Normalize(Cross(forward, right));
        const float aspect =
            static_cast<float>(m_swapchainExtent.width) / static_cast<float>(m_swapchainExtent.height);
        // Restart accumulation whenever the view or the analyzer moves.
        const ViewState view{position, forward, m_camera.PolarizerEnabled(),
                             m_camera.PolarizerAngleRadians(), m_camera.PolarizerEllipticityRadians()};
        if (m_lastView != view) m_accumulationResetRequested = true;
        m_lastView = view;

        const PushConstants constants{
            .cameraPositionFrame = {position.x, position.y, position.z, float(m_frameIndex)},
            .cameraForwardSamples = {forward.x, forward.y, forward.z, float(m_config.render.samplesPerPixel)},
            .cameraRightBounces = {right.x, right.y, right.z, m_accumulationResetRequested ? 1.0f : 0.0f},
            .cameraUpTanHalfFovY = {up.x, up.y, up.z, std::tan(m_config.camera.fovYDegrees * 0.5f * kPi / 180.0f)},
            .displayParams = {m_config.sky.exposure, aspect, 0.0f, 0.0f},
            .polarizer = {view.polarizerEnabled ? 1.0f : 0.0f, view.polarizerAngle, view.polarizerEllipticity, 0.0f},
            .imageSize = {m_swapchainExtent.width, m_swapchainExtent.height},
        };
        m_accumulationResetRequested = false;
        return constants;
    }

    void RecordCommandBuffer(const vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
    {
        using Stage = vk::PipelineStageFlagBits2;
        using Access = vk::AccessFlagBits2;
        const vk::Image image = m_swapchainImages[imageIndex];
        const vk::ImageSubresourceRange colorRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        // The compute pass overwrites every pixel, so the previous contents
        // (and layout) can always be discarded.
        const auto imageBarrier = [&](vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                                      vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess,
                                      vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
            const vk::ImageMemoryBarrier2 barrier{srcStage, srcAccess, dstStage, dstAccess, oldLayout, newLayout,
                                                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, colorRange};
            commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
        };
        const auto accumulationBarrier = [&](vk::AccessFlags2 srcAccess, vk::AccessFlags2 dstAccess) {
            const vk::BufferMemoryBarrier2 barrier{Stage::eComputeShader, srcAccess, Stage::eComputeShader, dstAccess,
                                                   VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                   *m_accumulationBuffer.buffer, 0, vk::WholeSize};
            commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setBufferMemoryBarriers(barrier));
        };

        commandBuffer.begin({});
        imageBarrier(Stage::eNone, Access::eNone, Stage::eComputeShader, Access::eShaderStorageWrite,
                     vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        accumulationBarrier(Access::eShaderStorageRead | Access::eShaderStorageWrite,
                            Access::eShaderStorageRead | Access::eShaderStorageWrite);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *m_computePipeline);
        using enum vk::DescriptorType;
        const vk::DescriptorImageInfo imageInfo{{}, *m_swapchainImageViews[imageIndex], vk::ImageLayout::eGeneral};
        const auto bufferInfo = [](const GpuBuffer& b) { return vk::DescriptorBufferInfo{*b.buffer, 0, vk::WholeSize}; };
        const std::array bufferInfos{bufferInfo(m_sceneDataBuffer), bufferInfo(m_mieScatteringBuffer),
                                     bufferInfo(m_rainbowScatteringBuffer), bufferInfo(m_accumulationBuffer)};
        const std::array writes{
            vk::WriteDescriptorSet{{}, 0, 0, eStorageImage, imageInfo},
            vk::WriteDescriptorSet{{}, 2, 0, eUniformBuffer, {}, bufferInfos[0]},
            vk::WriteDescriptorSet{{}, 7, 0, eStorageBuffer, {}, bufferInfos[1]},
            vk::WriteDescriptorSet{{}, 8, 0, eStorageBuffer, {}, bufferInfos[2]},
            vk::WriteDescriptorSet{{}, 9, 0, eStorageBuffer, {}, bufferInfos[3]},
        };
        commandBuffer.pushDescriptorSet(vk::PipelineBindPoint::eCompute, *m_pipelineLayout, 0, writes);
        commandBuffer.pushConstants<PushConstants>(*m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                                   BuildPushConstants());
        // 8x8 workgroups, matching the compute shader's local size; round up so
        // the whole render target is covered (the shader discards the overhang).
        constexpr uint32_t kTile = 8;
        const uint32_t groupsX = (m_swapchainExtent.width + kTile - 1) / kTile;
        const uint32_t groupsY = (m_swapchainExtent.height + kTile - 1) / kTile;
        commandBuffer.dispatch(groupsX, groupsY, 1);

        accumulationBarrier(Access::eShaderStorageWrite, Access::eShaderStorageRead);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *m_postProcessPipeline);
        commandBuffer.dispatch(groupsX, groupsY, 1);

        imageBarrier(Stage::eComputeShader, Access::eShaderStorageWrite,
                     Stage::eColorAttachmentOutput, Access::eColorAttachmentRead | Access::eColorAttachmentWrite,
                     vk::ImageLayout::eGeneral, vk::ImageLayout::eColorAttachmentOptimal);

        vk::RenderingAttachmentInfo colorAttachment{*m_swapchainImageViews[imageIndex],
                                                    vk::ImageLayout::eColorAttachmentOptimal};
        colorAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        commandBuffer.beginRendering(vk::RenderingInfo{{}, {{0, 0}, m_swapchainExtent}, 1, 0, colorAttachment});
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), static_cast<VkCommandBuffer>(*commandBuffer));
        commandBuffer.endRendering();

        imageBarrier(Stage::eColorAttachmentOutput, Access::eColorAttachmentWrite, Stage::eNone, Access::eNone,
                     vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);
        commandBuffer.end();
    }

    void RenderFrame()
    {
        FrameResources& frame = m_frames[m_currentFrame];
        const std::array fences{*frame.inFlight, *frame.presentDone};
        while (m_device.waitForFences(fences, VK_TRUE, UINT64_MAX) == vk::Result::eTimeout) {
        }

        const auto [acquireResult, imageIndex] = m_swapchain.acquireNextImage(UINT64_MAX, *frame.imageAvailable);
        if (acquireResult != vk::Result::eSuccess && acquireResult != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("Failed to acquire swapchain image.");
        }

        m_device.resetFences(fences);
        const vk::raii::CommandBuffer& commandBuffer = frame.commandBuffer;
        commandBuffer.reset();
        RecordCommandBuffer(commandBuffer, imageIndex);

        const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eComputeShader;
        vk::SubmitInfo submitInfo{};
        submitInfo.setWaitSemaphores(*frame.imageAvailable);
        submitInfo.setWaitDstStageMask(waitStage);
        submitInfo.setCommandBuffers(*commandBuffer);
        submitInfo.setSignalSemaphores(*frame.renderFinished);
        m_graphicsQueue.submit(submitInfo, *frame.inFlight);

        const vk::SwapchainPresentFenceInfoKHR presentFence{*frame.presentDone};
        vk::PresentInfoKHR presentInfo{};
        presentInfo.pNext = &presentFence;
        presentInfo.setWaitSemaphores(*frame.renderFinished);
        presentInfo.setSwapchains(*m_swapchain);
        presentInfo.setImageIndices(imageIndex);
        const vk::Result present = m_graphicsQueue.presentKHR(presentInfo);
        if (present != vk::Result::eSuccess && present != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("Failed to present swapchain image.");
        }

        ++m_frameIndex;
        m_currentFrame = (m_currentFrame + 1) % static_cast<uint32_t>(m_frames.size());
    }

    void MessageLoop()
    {
        using Clock = std::chrono::steady_clock;
        auto previousFrame = Clock::now();
        auto titleUpdate = previousFrame;
        uint32_t frames = 0;

        while (!glfwWindowShouldClose(m_window)) {
            glfwPollEvents();
            ProcessConfigChanges();

            const auto now = Clock::now();
            const double deltaSeconds = std::chrono::duration<double>(now - previousFrame).count();
            previousFrame = now;

            BuildGuiFrame();
            const ImGuiIO& io = ImGui::GetIO();
            if (!io.WantCaptureMouse && !io.WantCaptureKeyboard) {
                m_camera.Update(deltaSeconds, m_window, m_config);
            }
            RenderFrame();

            // Show the average FPS / frame time over the last second in the title.
            ++frames;
            const double elapsed = std::chrono::duration<double>(now - titleUpdate).count();
            if (elapsed >= 1.0) {
                const std::string title = std::format("Vulkan Path Tracer - {:.1f} FPS ({:.2f} ms)",
                                                      frames / elapsed, 1000.0 * elapsed / frames);
                glfwSetWindowTitle(m_window, title.c_str());
                frames = 0;
                titleUpdate = now;
            }
        }
    }

    // The present fence signals once presentation no longer needs
    // renderFinished, so the whole set can be reused together.
    struct FrameResources {
        vk::raii::CommandBuffer commandBuffer{nullptr};
        vk::raii::Semaphore imageAvailable{nullptr};
        vk::raii::Semaphore renderFinished{nullptr};
        vk::raii::Fence presentDone{nullptr};
        vk::raii::Fence inFlight{nullptr};
    };

    GLFWwindow* m_window = nullptr;
    bool m_imguiInitialized = false;
    bool m_showGui = true;

    // RAII Vulkan handles. Declaration order is destruction-reverse order:
    // device-child objects are declared after the device so they are destroyed
    // before it; the device after the instance; the instance last.
    vk::raii::Context m_context;
    vk::raii::Instance m_instance{nullptr};
    vk::raii::SurfaceKHR m_surface{nullptr};
    vk::raii::PhysicalDevice m_physicalDevice{nullptr};
    vk::raii::Device m_device{nullptr};
    // One queue does compute, ImGui drawing, and presentation.
    uint32_t m_queueFamily = 0;
    vk::raii::Queue m_graphicsQueue{nullptr};

    // Declared after the device and before the buffers, so the buffers are
    // freed first, then the allocator, then the device.
    vma::UniqueAllocator m_allocator;
    GpuBuffer m_sceneDataBuffer;
    GpuBuffer m_mieScatteringBuffer;
    GpuBuffer m_rainbowScatteringBuffer;
    GpuBuffer m_accumulationBuffer;

    vk::raii::SwapchainKHR m_swapchain{nullptr};
    vk::Format m_swapchainFormat = vk::Format::eUndefined;
    vk::Extent2D m_swapchainExtent{};
    std::vector<vk::Image> m_swapchainImages;
    std::vector<vk::raii::ImageView> m_swapchainImageViews;

    vk::raii::DescriptorSetLayout m_descriptorSetLayout{nullptr};
    vk::raii::PipelineLayout m_pipelineLayout{nullptr};
    vk::raii::Pipeline m_computePipeline{nullptr};
    vk::raii::Pipeline m_postProcessPipeline{nullptr};

    vk::raii::CommandPool m_commandPool{nullptr};
    std::vector<FrameResources> m_frames;
    uint32_t m_currentFrame = 0;
    uint64_t m_frameIndex = 0;
    bool m_accumulationResetRequested = true;
    // Camera + analyzer state of the last frame; any change restarts accumulation.
    struct ViewState {
        Vec3 position;
        Vec3 forward;
        bool polarizerEnabled;
        float polarizerAngle;
        float polarizerEllipticity;
        friend bool operator==(const ViewState&, const ViewState&) = default;
    };
    std::optional<ViewState> m_lastView;
    RuntimeConfig m_config{};
    ConfigFile m_configFile;
    bool m_cycleConfigRequested = false;
    bool m_saveConfigRequested = false;

    CameraController m_camera;
};

void RunVulkanPathTracer()
{
    VulkanPathTracer app;
    app.Run();
}
