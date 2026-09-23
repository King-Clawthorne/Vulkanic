#include "SkyTables.h"
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#ifndef __clang_analyzer__
#define VMA_IMPLEMENTATION
#endif
#include <vk_mem_alloc_raii.hpp>
#include <array>
#include <algorithm>
#include <chrono>
#include <format>
#include <print>
#include <cstddef>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

class CameraController {
public:
    struct View {
        float yaw, pitch;
        bool polarizerEnabled;
        float polarizerAngle, polarizerEllipticity;
        friend bool operator==(const View&, const View&) = default;
    };

    void Reset(const RuntimeConfig& config);
    void ClampPitch(const RuntimeConfig& config);
    void OnKeyPress(int key, const RuntimeConfig& config);
    void Update(double deltaSeconds, GLFWwindow* window, const RuntimeConfig& config);
    [[nodiscard]] View GetView() const;

private:
    float m_yaw = 0.0f, m_pitch = 0.0f;
    bool m_mouseLookActive = false;
    double m_lastMouseX = 0.0, m_lastMouseY = 0.0;
    bool m_polarizerEnabled = false, m_polarizerElliptical = false;
    float m_polarizerAngle = 0.0f, m_polarizerEllipticity = kPi * 0.125f;
};

namespace {
    float MaxPitchRadians(const RuntimeConfig& config) { return config.camera.maxPitchDegrees * kPi / 180.0f; }
    float Axis(GLFWwindow* window, int negativeKey, int positiveKey) {
        return static_cast<float>(glfwGetKey(window, positiveKey) == GLFW_PRESS) - static_cast<float>(glfwGetKey(window, negativeKey) == GLFW_PRESS);
    }
}

void CameraController::Reset(const RuntimeConfig& config) {
    const glm::vec3& from = config.camera.initialPosition;
    const glm::vec3& to = config.camera.initialLookAt;
    m_yaw = std::atan2(to.x - from.x, to.z - from.z);
    m_pitch = std::atan2(to.y - from.y, std::hypot(to.x - from.x, to.z - from.z));
}

void CameraController::ClampPitch(const RuntimeConfig& config) {
    m_pitch = std::clamp(m_pitch, -MaxPitchRadians(config), MaxPitchRadians(config));
}

void CameraController::OnKeyPress(int key, const RuntimeConfig& config) {
    if (key == GLFW_KEY_R) Reset(config);
    if (key == GLFW_KEY_P) m_polarizerEnabled = !m_polarizerEnabled;
    if (key == GLFW_KEY_C) m_polarizerElliptical = !m_polarizerElliptical;
}

void CameraController::Update(double deltaSeconds, GLFWwindow* window, const RuntimeConfig& config) {
    if (glfwGetWindowAttrib(window, GLFW_FOCUSED) != GLFW_TRUE) {
        m_mouseLookActive = false;
        return;
    }
    
    const float dt = static_cast<float>(std::min(deltaSeconds, 0.1));

    double cursorX = 0.0, cursorY = 0.0;
    glfwGetCursorPos(window, &cursorX, &cursorY);
    const bool mouseLook = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (mouseLook && m_mouseLookActive) {
        m_yaw += static_cast<float>(cursorX - m_lastMouseX) * config.input.mouseSensitivity;
        m_pitch -= static_cast<float>(cursorY - m_lastMouseY) * config.input.mouseSensitivity;
    }
    
    m_mouseLookActive = mouseLook;
    m_lastMouseX = cursorX;
    m_lastMouseY = cursorY;
    ClampPitch(config);

    const float analyzer = Axis(window, GLFW_KEY_LEFT_BRACKET, GLFW_KEY_RIGHT_BRACKET) * config.input.polarizerRotateSpeed * dt;
    if (m_polarizerElliptical) m_polarizerEllipticity = std::clamp(m_polarizerEllipticity + analyzer, -kPi * 0.25f, kPi * 0.25f);
    else m_polarizerAngle += analyzer;
}

CameraController::View CameraController::GetView() const {
    return {.yaw = m_yaw, .pitch = m_pitch, .polarizerEnabled = m_polarizerEnabled, .polarizerAngle = m_polarizerAngle, .polarizerEllipticity = m_polarizerElliptical ? m_polarizerEllipticity : 0.0f};
}

constexpr uint32_t kFramesInFlight = 2;
constexpr uint32_t kPathTracerSpirv[] = {
#include "pathTracer.comp.spv.inc"
};

struct alignas(16) SceneData {
    glm::vec4 skySpectralParams, skyRadiiScaleHeights, skySunDirectionRadius;
    glm::uvec4 skySampleCounts;
    glm::vec4 skyVrtParams, rainbowCenterEnabled, rainbowRadiiEdge, rainbowOptical;
    glm::uvec4 rainbowMultiple;
    glm::vec4 spectralBands[kSpectralBandCount], cieXyz[kSpectralBandCount], sunDisk;
    glm::vec4 mieBands[kSpectralBandCount], rainbowAxisX, rainbowAxisZ, apparentSun;
};

static_assert(offsetof(SceneData, skyVrtParams) == 64);
static_assert(offsetof(SceneData, rainbowMultiple) == 128);
static_assert(offsetof(SceneData, spectralBands) == 144);
static_assert(sizeof(SceneData) == 1024);

constexpr std::array kDescriptorTypes{vk::DescriptorType::eStorageImage,  vk::DescriptorType::eUniformBuffer,
                                      vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageBuffer,
                                      vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageBuffer};

struct PushConstants {
    glm::vec4 forward, right, up, frame, polarizer;
    glm::uvec2 imageSize;
};

class VulkanPathTracer {
public:
    ~VulkanPathTracer() {
        if (*m_device) vkDeviceWaitIdle(*m_device);
        if (m_window != nullptr) {
            glfwDestroyWindow(m_window);
            m_window = nullptr;
        }
        
        glfwTerminate();
    }

    void Run() {
        m_camera.Reset(m_config);
        CreateWindowAndShow();
        CreateInstance();
        CreateSurface();
        PickPhysicalDevice();
        CreateLogicalDevice();
        m_allocator = vma::raii::Allocator(m_instance, m_device, vma::AllocatorCreateInfo{}.setPhysicalDevice(*m_physicalDevice).setVulkanApiVersion(VK_API_VERSION_1_4));
        m_commandPool = vk::raii::CommandPool(m_device, {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, m_queueFamily});
        CreateSceneBuffers();
        CreateSwapchain();
        CreateDescriptorSetLayout();
        CreatePipeline();
        CreateFrameResources();
        MessageLoop();
        m_device.waitIdle();
    }

private:
    static void KeyCallback(GLFWwindow* window, int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods) {
        auto* app = static_cast<VulkanPathTracer*>(glfwGetWindowUserPointer(window));
        if (app == nullptr || action != GLFW_PRESS) return;
        app->m_camera.OnKeyPress(key, app->m_config);
    }

    [[nodiscard]] SceneData BuildSceneData() const {
        const SkySpectralConfig& s = m_config.sky.spectral;
        const RainbowConfig& r = m_config.rainbow;
        SceneData sceneData{
            .skySpectralParams = {s.betaRayleigh550, s.aerosols[0].beta, 0.0f, s.sunRadiance550},
            .skyRadiiScaleHeights = {s.earthRadius, s.atmosphereRadius, s.scaleHeightRayleigh, s.aerosols[0].scaleHeight},
            .skySunDirectionRadius = {s.sunDirection[0], s.sunDirection[1], s.sunDirection[2], s.sunRadius},
            .skySampleCounts = {s.secondarySamples, s.viewSteps, s.samples, s.scatteringOrders},
            .skyVrtParams = {s.sunAa, s.rayleighDepolarization, static_cast<float>(s.mieTableAngleBins), s.aerosols[1].scaleHeight},
            .rainbowRadiiEdge = {r.radii.x, r.radii.y, r.radii.z, r.edgeSoftness},
            .rainbowOptical = {r.scatteringCoefficient, r.extinctionCoefficient, static_cast<float>(r.angleBins), static_cast<float>(r.viewSteps)},
            .rainbowMultiple = {r.scatteringOrders, r.multipleScatteringSamples, r.multipleScatteringSteps, 0},
        };
        
        float ySum = 0.0f;
        const double sun550 = ComputeSpectralBand(6).sunIrradianceScale;
        for (int band = 0; band < kSpectralBandCount; ++band) {
            const SpectralBand spectral = ComputeSpectralBand(band);
            sceneData.spectralBands[band][0] = static_cast<float>(s.betaRayleigh550 * spectral.betaRayleighScale);
            sceneData.spectralBands[band][1] = static_cast<float>(s.sunRadiance550 * spectral.sunIrradianceScale / sun550);
            sceneData.spectralBands[band][2] = static_cast<float>(kSpectralLambdaMinNm + kSpectralLambdaStepNm * band);
            sceneData.spectralBands[band][3] = static_cast<float>(spectral.ozoneCrossSection * s.ozoneDobsonUnits * 2.687e20 / 15000.0);
            for (size_t i = 0; i < 3; ++i) sceneData.cieXyz[band][i] = static_cast<float>(spectral.cie[i]);
            ySum += sceneData.cieXyz[band][1];
        }
        
        for (auto& xyz : sceneData.cieXyz)
            for (int i = 0; i < 3; ++i) xyz[i] /= ySum;
        
        sceneData.sunDisk[0] = 2.0f * kPi * (1.0f - std::cos(s.sunRadius));
        sceneData.sunDisk[1] = std::cos(s.sunRadius + s.sunAa);
        sceneData.sunDisk[2] = std::cos(s.sunRadius - s.sunAa);
        const float sunNorm = std::hypot(s.sunDirection[0], s.sunDirection[1], s.sunDirection[2]);
        const float trueAltitude = std::asin(s.sunDirection[1] / sunNorm) * 180.0f / kPi;
        const auto refraction = [](float altitude) {
            const float h = std::max(altitude, -1.0f);
            return 1.02f / 60.0f / std::tan((h + 10.3f / (h + 5.11f)) * kPi / 180.0f);
        };
        
        const float apparentAltitude = (trueAltitude + refraction(trueAltitude)) * kPi / 180.0f;
        const float horizontal = std::hypot(s.sunDirection[0], s.sunDirection[2]);
        sceneData.apparentSun[0] = std::cos(apparentAltitude) * s.sunDirection[0] / horizontal;
        sceneData.apparentSun[1] = std::sin(apparentAltitude);
        sceneData.apparentSun[2] = std::cos(apparentAltitude) * s.sunDirection[2] / horizontal;
        sceneData.apparentSun[3] = 1.0f + ((refraction(trueAltitude + 0.1f) - refraction(trueAltitude - 0.1f)) / 0.2f);
        const float sunLength = std::hypot(s.sunDirection[0], s.sunDirection[2]);
        const float ax = -s.sunDirection[0] / sunLength;
        const float az = -s.sunDirection[2] / sunLength;
        sceneData.rainbowCenterEnabled[0] = ax * r.distance;
        sceneData.rainbowCenterEnabled[1] = r.height;
        sceneData.rainbowCenterEnabled[2] = az * r.distance;
        sceneData.rainbowCenterEnabled[3] = static_cast<float>(r.enabled);
        sceneData.rainbowAxisX[0] = az;
        sceneData.rainbowAxisX[2] = -ax;
        sceneData.rainbowAxisZ[0] = ax;
        sceneData.rainbowAxisZ[2] = az;
        
        for (size_t aerosol = 0; aerosol < 2; ++aerosol) {
            const double beta = s.aerosols[aerosol].beta;
            const double reference = m_mieCrossSections[aerosol][6].x;
            for (int band = 0; band < kSpectralBandCount; ++band) {
                const glm::dvec2& cross = m_mieCrossSections[aerosol][static_cast<size_t>(band)];
                sceneData.mieBands[band][2 * aerosol] = static_cast<float>(beta * cross.x / reference);
                sceneData.mieBands[band][2 * aerosol + 1] = static_cast<float>(beta * cross.y / reference);
            }
        }
        
        return sceneData;
    }

    template <typename T>
    [[nodiscard]] vma::raii::Buffer CreateTableBuffer(const std::vector<T>& table) const {
        vma::raii::Buffer buffer = CreateBuffer(table.size() * sizeof(T), vk::BufferUsageFlagBits::eStorageBuffer);
        UploadToBuffer(buffer, std::as_bytes(std::span{table}));
        return buffer;
    }

    void CreateSceneBuffers() {
        const SkySpectralConfig& s = m_config.sky.spectral;
        m_sceneDataBuffer = CreateBuffer(sizeof(SceneData), vk::BufferUsageFlagBits::eUniformBuffer);
        std::vector<glm::vec4> mie = ComputeMieScatteringTable(s, 0, m_mieCrossSections[0]);
        std::ranges::copy(ComputeMieScatteringTable(s, 1, m_mieCrossSections[1]), std::back_inserter(mie));
        const int bins = std::max(2, static_cast<int>(s.mieTableAngleBins));
        AppendSamplingCdf(mie, bins, 0);
        AppendSamplingCdf(mie, bins, static_cast<size_t>(kSpectralBandCount) * static_cast<size_t>(bins));
        m_mieScatteringBuffer = CreateTableBuffer(mie);
        std::vector<glm::vec4> rainbow = ComputeRainbowScatteringTable(m_config.rainbow, s.sunRadius);
        AppendSamplingCdf(rainbow, static_cast<int>(m_config.rainbow.angleBins), 0);
        m_rainbowScatteringBuffer = CreateTableBuffer(rainbow);
        m_transmittanceBuffer = CreateTableBuffer(ComputeTransmittanceTable(s));
        const SceneData sceneData = BuildSceneData();
        UploadToBuffer(m_sceneDataBuffer, std::as_bytes(std::span{&sceneData, 1}));
    }

    void CreateWindowAndShow() {
        if (glfwInit() != GLFW_TRUE) throw std::runtime_error("GLFW initialization failed.");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        m_window = glfwCreateWindow(static_cast<int>(m_config.render.width), static_cast<int>(m_config.render.height), "Vulkanic", nullptr, nullptr);
        if (m_window == nullptr) throw std::runtime_error("GLFW could not create the application window.");
        
        glfwSetWindowUserPointer(m_window, this);
        glfwSetKeyCallback(m_window, KeyCallback);
        std::println("Controls: right-drag to look, R resets the view.");
        std::println("Polarizer: P toggles it, C switches linear/elliptical, [ ] adjusts it.");
    }

    void CreateInstance() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        if (glfwExtensions == nullptr || glfwExtensionCount == 0)
            throw std::runtime_error("GLFW did not provide the required Vulkan instance extensions.");
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        extensions.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
        extensions.push_back(VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
        vk::ApplicationInfo appInfo{"Vulkan Path Tracer", 1, "None", 1, VK_API_VERSION_1_4};
        vk::InstanceCreateInfo createInfo{};
        createInfo.setPApplicationInfo(&appInfo).setPEnabledExtensionNames(extensions);
        m_instance = vk::raii::Instance(m_context, createInfo);
    }

    void CreateSurface() {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        const VkResult result = glfwCreateWindowSurface(static_cast<VkInstance>(*m_instance), m_window, nullptr, &surface);
        if (result != VK_SUCCESS)
            throw std::runtime_error(std::format("GLFW failed to create the Vulkan surface (VkResult {}).", static_cast<int>(result)));
        m_surface = vk::raii::SurfaceKHR(m_instance, surface);
    }

    void PickPhysicalDevice() {
        constexpr std::array requiredExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        for (const auto& candidate : m_instance.enumeratePhysicalDevices()) {
            if (candidate.getProperties().apiVersion < VK_API_VERSION_1_4) continue;

            const auto extensions = candidate.enumerateDeviceExtensionProperties();
            const bool hasExtensions = std::ranges::all_of(requiredExtensions, [&](const char* required) {
                return std::ranges::any_of(extensions, [&](const vk::ExtensionProperties& available) {
                    return std::strcmp(available.extensionName, required) == 0;
                });
            });
            if (!hasExtensions) continue;

            const auto features = candidate.getFeatures2<vk::PhysicalDeviceFeatures2,
                                                         vk::PhysicalDeviceVulkan13Features,
                                                         vk::PhysicalDeviceVulkan14Features>();
            if (!features.get<vk::PhysicalDeviceFeatures2>().features.shaderStorageImageWriteWithoutFormat ||
                !features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 ||
                !features.get<vk::PhysicalDeviceVulkan14Features>().pushDescriptor) continue;

            const auto families = candidate.getQueueFamilyProperties();
            for (uint32_t i = 0; i < families.size(); ++i) {
                if (!(families[i].queueFlags & vk::QueueFlagBits::eGraphics) ||
                    !candidate.getSurfaceSupportKHR(i, *m_surface)) continue;
                const auto capabilities = candidate.getSurfaceCapabilitiesKHR(*m_surface);
                const auto formats = candidate.getSurfaceFormatsKHR(*m_surface);
                const auto modes = candidate.getSurfacePresentModesKHR(*m_surface);
                if (!(capabilities.supportedUsageFlags & vk::ImageUsageFlagBits::eStorage) || formats.empty() || modes.empty()) continue;
                m_physicalDevice = candidate;
                m_queueFamily = i;
                return;
            }
        }
        throw std::runtime_error("No Vulkan 1.4 device supports the required features, extensions, graphics, and presentation.");
    }

    void CreateLogicalDevice() {
        constexpr std::array extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        vk::PhysicalDeviceFeatures features{};
        features.shaderStorageImageWriteWithoutFormat = true;
        vk::PhysicalDeviceVulkan13Features features13{};
        features13.synchronization2 = true;
        vk::PhysicalDeviceVulkan14Features features14{};
        features14.pushDescriptor = true;
        features13.setPNext(&features14);
        const float priority = 1.0f;
        vk::DeviceQueueCreateInfo queueInfo{};
        queueInfo.setQueueFamilyIndex(m_queueFamily).setQueuePriorities(priority);
        vk::DeviceCreateInfo createInfo{};
        createInfo.setPNext(&features13).setPEnabledFeatures(&features).setQueueCreateInfos(queueInfo).setPEnabledExtensionNames(extensions);
        m_device = vk::raii::Device(m_physicalDevice, createInfo);
        m_graphicsQueue = m_device.getQueue(m_queueFamily, 0);
    }

    [[nodiscard]] vma::raii::Buffer CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                                 vma::AllocationCreateFlags flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite) const {
        return vma::raii::Buffer(m_allocator, vk::BufferCreateInfo{{}, size, usage}, vma::AllocationCreateInfo{flags, vma::MemoryUsage::eAuto});
    }

    static void UploadToBuffer(const vma::raii::Buffer& buffer, std::span<const std::byte> data) {
        buffer.getAllocation().copyFromMemory(data.data(), 0, data.size());
    }

    void CreateSwapchain() {
        const auto capabilities = m_physicalDevice.getSurfaceCapabilitiesKHR(*m_surface);
        const auto formats = m_physicalDevice.getSurfaceFormatsKHR(*m_surface);
        const auto modes = m_physicalDevice.getSurfacePresentModesKHR(*m_surface);
        const vk::SurfaceFormatKHR desiredFormat{vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
        const auto formatIt = std::ranges::find(formats, desiredFormat);
        if (formatIt == formats.end()) throw std::runtime_error("Surface does not support B8G8R8A8_UNORM with SRGB_NONLINEAR.");
        const vk::PresentModeKHR presentMode = std::ranges::contains(modes, vk::PresentModeKHR::eImmediate)
                                                   ? vk::PresentModeKHR::eImmediate
                                                   : vk::PresentModeKHR::eMailbox;
        if (!std::ranges::contains(modes, presentMode)) throw std::runtime_error("Surface supports neither immediate nor mailbox presentation.");
        vk::Extent2D extent{m_config.render.width, m_config.render.height};
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            extent = capabilities.currentExtent;
        } else {
            extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        }
        const uint32_t imageCount = std::max(kFramesInFlight, capabilities.minImageCount);
        vk::SwapchainCreateInfoKHR createInfo{};
        createInfo.setSurface(*m_surface).setMinImageCount(imageCount).setImageFormat(desiredFormat.format).setImageColorSpace(desiredFormat.colorSpace)
            .setImageExtent(extent).setImageArrayLayers(1).setImageUsage(vk::ImageUsageFlagBits::eStorage)
            .setImageSharingMode(vk::SharingMode::eExclusive).setPreTransform(capabilities.currentTransform)
            .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque).setPresentMode(presentMode).setClipped(true);
        m_swapchain = vk::raii::SwapchainKHR(m_device, createInfo);
        m_swapchainExtent = extent;
        m_swapchainImages = m_swapchain.getImages();
        m_swapchainImageViews.clear();
        m_renderFinished.clear();

        for (const auto image : m_swapchainImages) {
            m_swapchainImageViews.emplace_back(m_device, vk::ImageViewCreateInfo{{}, image, vk::ImageViewType::e2D,
                desiredFormat.format, {}, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});
            m_renderFinished.emplace_back(m_device.createSemaphore({}));
        }
    }

    void CreateDescriptorSetLayout() {
        std::array<vk::DescriptorSetLayoutBinding, kDescriptorTypes.size()> layoutBindings{};
        for (uint32_t i = 0; i < layoutBindings.size(); ++i) {
            layoutBindings[i] = {i, kDescriptorTypes[i], 1, vk::ShaderStageFlagBits::eCompute};
        }
        
        vk::DescriptorSetLayoutCreateInfo createInfo{vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor};
        createInfo.setBindings(layoutBindings);
        m_descriptorSetLayout = vk::raii::DescriptorSetLayout(m_device, createInfo);
    }

    void CreatePipeline() {
        const vk::raii::ShaderModule computeModule{m_device, vk::ShaderModuleCreateInfo{{}, sizeof(kPathTracerSpirv), std::data(kPathTracerSpirv)}};

        const std::array<uint32_t, 5> settings = {std::max(1u, m_config.sky.spectral.scatteringOrders),
                                                  m_config.sky.spectral.viewSteps, m_config.sky.spectral.samples,
                                                  m_config.sky.spectral.secondarySamples, (m_config.rainbow.enabled != 0u) ? 1u : 0u};

        const std::array<vk::SpecializationMapEntry, 5> entries = {{{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}, {4, 16, 4}}};
        const vk::SpecializationInfo specialization{static_cast<uint32_t>(entries.size()), entries.data(), sizeof(settings), settings.data()};
        const vk::PushConstantRange pushRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants)};
        vk::PipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.setSetLayouts(*m_descriptorSetLayout);
        layoutInfo.setPushConstantRanges(pushRange);
        m_pipelineLayout = vk::raii::PipelineLayout(m_device, layoutInfo);
        const vk::PipelineShaderStageCreateInfo stage{{}, vk::ShaderStageFlagBits::eCompute, *computeModule, "main", &specialization};
        m_computePipeline = vk::raii::Pipeline(m_device, nullptr, vk::ComputePipelineCreateInfo{{}, stage, *m_pipelineLayout});
    }

    void CreateFrameResources() {
        const uint32_t count = kFramesInFlight;
        vk::raii::CommandBuffers commandBuffers(m_device, {*m_commandPool, vk::CommandBufferLevel::ePrimary, count});
        const vk::FenceCreateInfo signaled{vk::FenceCreateFlagBits::eSignaled};
        m_frames.clear();

        for (uint32_t i = 0; i < count; ++i) {
            m_frames.push_back({.commandBuffer = std::move(commandBuffers[i]), .imageAvailable = m_device.createSemaphore({}), .inFlight = m_device.createFence(signaled)});
        }

        const vk::DeviceSize pixels = static_cast<vk::DeviceSize>(m_swapchainExtent.width) * m_swapchainExtent.height;
        m_accumulationBuffer = CreateBuffer(pixels * 4 * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer, {});
    }

    PushConstants BuildPushConstants() {
        const CameraController::View view = m_camera.GetView();
        const bool reset = m_lastView != view;
        m_lastView = view;

        const float tanHalfFov = std::tan(m_config.camera.fovYDegrees * kPi / 360.0f);
        const float aspect = static_cast<float>(m_swapchainExtent.width) / static_cast<float>(m_swapchainExtent.height);
        const float fx = std::sin(view.yaw) * std::cos(view.pitch);
        const float fy = std::sin(view.pitch);
        const float fz = std::cos(view.yaw) * std::cos(view.pitch);
        const float rx = std::cos(view.yaw);
        const float rz = -std::sin(view.yaw);

        const PushConstants constants{
            .forward = {fx, fy, fz, static_cast<float>(m_config.render.samplesPerPixel)},
            .right = {rx * aspect * tanHalfFov, 0.0f, rz * aspect * tanHalfFov, 0.0f},
            .up = {fy * rz * tanHalfFov, (fz * rx - fx * rz) * tanHalfFov, -fy * rx * tanHalfFov, 0.0f},
            .frame = {static_cast<float>(m_frameIndex), m_config.sky.exposure, reset ? 1.0f : 0.0f, 0.0f},
            .polarizer = {view.polarizerEnabled ? 1.0f : 0.0f, view.polarizerAngle, view.polarizerEllipticity, 0.0f},
            .imageSize = {m_swapchainExtent.width, m_swapchainExtent.height},
        };

        return constants;
    }

    void RecordCommandBuffer(const vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex) {
        using Stage = vk::PipelineStageFlagBits2;
        using Access = vk::AccessFlagBits2;
        const vk::Image image = m_swapchainImages[imageIndex];
        const vk::ImageSubresourceRange colorRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        const auto imageBarrier = [&](vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                                      vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess,
                                      vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
            const vk::ImageMemoryBarrier2 barrier{srcStage, srcAccess, dstStage, dstAccess, oldLayout, newLayout,
                                                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, colorRange};
            commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
        };

        commandBuffer.begin({});
        const vk::MemoryBarrier2 accumulation{Stage::eComputeShader, Access::eShaderStorageWrite, Stage::eComputeShader, Access::eShaderStorageRead | Access::eShaderStorageWrite};
        commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setMemoryBarriers(accumulation));
        imageBarrier(Stage::eNone, Access::eNone, Stage::eComputeShader, Access::eShaderStorageWrite,
                     vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *m_computePipeline);
        const vk::DescriptorImageInfo imageInfo{{}, *m_swapchainImageViews[imageIndex], vk::ImageLayout::eGeneral};
        const auto bufferInfo = [](const vma::raii::Buffer& b) { return vk::DescriptorBufferInfo{*b, 0, vk::WholeSize}; };
        const std::array bufferInfos{bufferInfo(m_sceneDataBuffer), bufferInfo(m_mieScatteringBuffer),
                                     bufferInfo(m_rainbowScatteringBuffer), bufferInfo(m_transmittanceBuffer),
                                     bufferInfo(m_accumulationBuffer)};

        std::array<vk::WriteDescriptorSet, kDescriptorTypes.size()> writes{};
        writes[0] = vk::WriteDescriptorSet{{}, 0, 0, kDescriptorTypes[0], imageInfo};
        for (uint32_t i = 1; i < writes.size(); ++i) {
            writes[i] = vk::WriteDescriptorSet{{}, i, 0, kDescriptorTypes[i], {}, bufferInfos[i - 1]};
        }

        commandBuffer.pushDescriptorSet(vk::PipelineBindPoint::eCompute, *m_pipelineLayout, 0, writes);
        commandBuffer.pushConstants<PushConstants>(*m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                                   BuildPushConstants());

        constexpr uint32_t kTile = 8;
        const uint32_t groupsX = (m_swapchainExtent.width + kTile - 1) / kTile;
        const uint32_t groupsY = (m_swapchainExtent.height + kTile - 1) / kTile;
        commandBuffer.dispatch(groupsX, groupsY, 1);

        imageBarrier(Stage::eComputeShader, Access::eShaderStorageWrite, Stage::eNone, Access::eNone,
                     vk::ImageLayout::eGeneral, vk::ImageLayout::ePresentSrcKHR);

        commandBuffer.end();
    }

    void RenderFrame() {
        FrameResources& frame = m_frames[m_currentFrame];
        while (m_device.waitForFences(*frame.inFlight, VK_TRUE, UINT64_MAX) == vk::Result::eTimeout) {}
        const uint32_t imageIndex = m_swapchain.acquireNextImage(UINT64_MAX, *frame.imageAvailable).value;

        m_device.resetFences(*frame.inFlight);
        const vk::raii::CommandBuffer& commandBuffer = frame.commandBuffer;
        commandBuffer.reset();
        RecordCommandBuffer(commandBuffer, imageIndex);

        const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eComputeShader;
        vk::SubmitInfo submitInfo{};
        submitInfo.setWaitSemaphores(*frame.imageAvailable);
        submitInfo.setWaitDstStageMask(waitStage);
        submitInfo.setCommandBuffers(*commandBuffer);
        submitInfo.setSignalSemaphores(*m_renderFinished[imageIndex]);
        m_graphicsQueue.submit(submitInfo, *frame.inFlight);

        vk::PresentInfoKHR presentInfo{};
        presentInfo.setWaitSemaphores(*m_renderFinished[imageIndex]);
        presentInfo.setSwapchains(*m_swapchain);
        presentInfo.setImageIndices(imageIndex);
        std::ignore = m_graphicsQueue.presentKHR(presentInfo);

        ++m_frameIndex;
        m_currentFrame = (m_currentFrame + 1) % static_cast<uint32_t>(m_frames.size());
    }

    void MessageLoop() {
        auto previousFrame = std::chrono::steady_clock::now();
        auto titleUpdate = previousFrame;
        uint32_t frames = 0;

        while (glfwWindowShouldClose(m_window) == 0) {
            glfwPollEvents();

            const auto now = std::chrono::steady_clock::now();
            const double deltaSeconds = std::chrono::duration<double>(now - previousFrame).count();
            previousFrame = now;

            m_camera.Update(deltaSeconds, m_window, m_config);
            RenderFrame();

            ++frames;
            const double elapsed = std::chrono::duration<double>(now - titleUpdate).count();
            if (elapsed >= 1.0) {
                const std::string title = std::format("Vulkanic - {:.1f} FPS ({:.2f} ms)", frames / elapsed, 1000.0 * elapsed / frames);
                glfwSetWindowTitle(m_window, title.c_str());
                frames = 0;
                titleUpdate = now;
            }
        }
    }

    struct FrameResources {
        vk::raii::CommandBuffer commandBuffer{nullptr};
        vk::raii::Semaphore imageAvailable{nullptr};
        vk::raii::Fence inFlight{nullptr};
    };

    GLFWwindow* m_window = nullptr;

    vk::raii::Context m_context;
    vk::raii::Instance m_instance{nullptr};
    vk::raii::SurfaceKHR m_surface{nullptr};
    vk::raii::PhysicalDevice m_physicalDevice{nullptr};
    vk::raii::Device m_device{nullptr};
    uint32_t m_queueFamily = 0;
    vk::raii::Queue m_graphicsQueue{nullptr};

    vma::raii::Allocator m_allocator{nullptr};
    vma::raii::Buffer m_sceneDataBuffer{nullptr}, m_mieScatteringBuffer{nullptr}, m_rainbowScatteringBuffer{nullptr};
    vma::raii::Buffer m_transmittanceBuffer{nullptr}, m_accumulationBuffer{nullptr};
    std::array<std::array<glm::dvec2, kSpectralBandCount>, 2> m_mieCrossSections{};

    vk::raii::SwapchainKHR m_swapchain{nullptr};
    vk::Extent2D m_swapchainExtent{};
    std::vector<vk::Image> m_swapchainImages;
    std::vector<vk::raii::ImageView> m_swapchainImageViews;
    std::vector<vk::raii::Semaphore> m_renderFinished;

    vk::raii::DescriptorSetLayout m_descriptorSetLayout{nullptr};
    vk::raii::PipelineLayout m_pipelineLayout{nullptr};
    vk::raii::Pipeline m_computePipeline{nullptr};
    vk::raii::CommandPool m_commandPool{nullptr};

    std::vector<FrameResources> m_frames;
    uint32_t m_currentFrame = 0;
    uint64_t m_frameIndex = 0;
    RuntimeConfig m_config{};
    CameraController m_camera;
    std::optional<CameraController::View> m_lastView;
};

int main() {
    try {
        VulkanPathTracer app;
        app.Run();
    } catch (const std::exception& e) {
        std::fputs(e.what(), stderr);
        return 1;
    }
}
