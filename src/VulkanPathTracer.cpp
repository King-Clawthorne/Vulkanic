#include "SkyTables.h"

#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc_raii.hpp>
#include <array>
#include <algorithm>
#include <chrono>
#include <format>
#include <print>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <vector>
#include <cmath>

class CameraController {
public:
    struct View {
        float yaw;
        float pitch;
        bool polarizerEnabled;
        float polarizerAngle;
        float polarizerEllipticity;
    };

    void Reset(const RuntimeConfig& config);
    void ClampPitch(const RuntimeConfig& config);

    void OnKeyPress(int key, const RuntimeConfig& config);

    void Update(double deltaSeconds, GLFWwindow* window, const RuntimeConfig& config);

    [[nodiscard]] View GetView() const;

private:
    float m_yaw = 0.0f;
    float m_pitch = 0.0f;

    bool m_mouseLookActive = false;
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;

    bool m_polarizerEnabled = false;
    bool m_polarizerElliptical = false;
    float m_polarizerAngle = 0.0f;
    float m_polarizerEllipticity = kPi * 0.125f;
};

namespace {
    float MaxPitchRadians(const RuntimeConfig& config) { return config.camera.maxPitchDegrees * kPi / 180.0f; }

    float Axis(GLFWwindow* window, int negativeKey, int positiveKey) {
        return float(glfwGetKey(window, positiveKey) == GLFW_PRESS) - float(glfwGetKey(window, negativeKey) == GLFW_PRESS);
    }
}

void CameraController::Reset(const RuntimeConfig& config) {
    const Vec3& from = config.camera.initialPosition;
    const Vec3& to = config.camera.initialLookAt;
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
        m_yaw += float(cursorX - m_lastMouseX) * config.input.mouseSensitivity;
        m_pitch -= float(cursorY - m_lastMouseY) * config.input.mouseSensitivity;
    }
    m_mouseLookActive = mouseLook;
    m_lastMouseX = cursorX;
    m_lastMouseY = cursorY;

    ClampPitch(config);

    const float analyzer = Axis(window, GLFW_KEY_LEFT_BRACKET, GLFW_KEY_RIGHT_BRACKET) * config.input.polarizerRotateSpeed * dt;
    if (m_polarizerElliptical) {
        m_polarizerEllipticity = std::clamp(m_polarizerEllipticity + analyzer, -kPi * 0.25f, kPi * 0.25f);
    } else {
        m_polarizerAngle += analyzer;
    }
}

CameraController::View CameraController::GetView() const {
    return {m_yaw, m_pitch, m_polarizerEnabled, m_polarizerAngle, m_polarizerElliptical ? m_polarizerEllipticity : 0.0f};
}

constexpr uint32_t kFramesInFlight = 2;

constexpr uint32_t kPathTracerSpirv[] = {
#include "pathTracer.comp.spv.inc"
};

struct alignas(16) SceneData {
    float skySpectralParams[4];
    float skyRadiiScaleHeights[4];
    float skySunDirectionRadius[4];
    uint32_t skySampleCounts[4];
    float skyVrtParams[4];
    float rainbowCenterEnabled[4];
    float rainbowRadiiEdge[4];
    float rainbowOptical[4];
    uint32_t rainbowMultiple[4];
    float spectralBands[kSpectralBandCount][4];
    float cieXyz[kSpectralBandCount][4];
};

struct PushConstants {
    float forward[4];
    float right[4];
    float up[4];
    float frame[4];
    float polarizer[4];
    uint32_t imageSize[2];
};

class VulkanPathTracer {
public:
    ~VulkanPathTracer() {
        if (*m_device) {
            m_device.waitIdle();
        }
        if (m_window != nullptr) {
            glfwDestroyWindow(m_window);
            m_window = nullptr;
        }
        glfwTerminate();
    }

    void Run() {
        m_camera.Reset(m_config);
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
        MessageLoop();
        m_device.waitIdle();
    }

private:
    static void KeyCallback(GLFWwindow* window, int key, int, int action, int) {
        auto* app = static_cast<VulkanPathTracer*>(glfwGetWindowUserPointer(window));
        if (app == nullptr || action != GLFW_PRESS) {
            return;
        }
        app->m_camera.OnKeyPress(key, app->m_config);
    }

    SceneData BuildSceneData() const {
        const SkySpectralConfig& s = m_config.sky.spectral;
        const RainbowConfig& r = m_config.rainbow;
        SceneData sceneData{
            .skySpectralParams = {s.betaRayleigh550, s.betaMie, s.sunTemperatureKelvin, s.sunRadiance550},
            .skyRadiiScaleHeights = {s.earthRadius, s.atmosphereRadius, s.scaleHeightRayleigh, s.scaleHeightMie},
            .skySunDirectionRadius = {s.sunDirection[0], s.sunDirection[1], s.sunDirection[2], s.sunRadius},
            .skySampleCounts = {s.secondarySamples, s.viewSteps, s.samples, s.scatteringOrders},
            .skyVrtParams = {s.sunAa, s.rayleighDepolarization, float(s.mieTableAngleBins), 0.0f},
            .rainbowCenterEnabled = {r.center.x, r.center.y, r.center.z, float(r.enabled)},
            .rainbowRadiiEdge = {r.radii.x, r.radii.y, r.radii.z, r.edgeSoftness},
            .rainbowOptical = {r.scatteringCoefficient, r.extinctionCoefficient, float(r.angleBins), float(r.viewSteps)},
            .rainbowMultiple = {r.scatteringOrders, r.multipleScatteringSamples, r.multipleScatteringSteps, 0},
        };
        float ySum = 0.0f;
        for (int band = 0; band < kSpectralBandCount; ++band) {
            const float wavelength = float(kSpectralLambdaMinNm + kSpectralLambdaStepNm * band);
            const float ratio = 550.0f / wavelength;
            const float lambda = wavelength * 1.0e-9f;
            constexpr float reference = 550.0e-9f;
            constexpr float c2 = 1.4387769e-2f;
            const float shape = std::pow(reference / lambda, 5.0f) * (std::exp(c2 / (reference * s.sunTemperatureKelvin)) - 1.0f) / (std::exp(c2 / (lambda * s.sunTemperatureKelvin)) - 1.0f);
            sceneData.spectralBands[band][0] = s.betaRayleigh550 * ratio * ratio * ratio * ratio;
            sceneData.spectralBands[band][1] = s.sunRadiance550 * shape;
            const auto g = [&](float mean, float left, float right) {
                const float x = (wavelength - mean) * (wavelength < mean ? left : right);
                return std::exp(-0.5f * x * x);
            };
            float* xyz = sceneData.cieXyz[band];
            xyz[0] = std::max(0.0f, 1.056f * g(599.8f, 0.0264f, 0.0323f) + 0.362f * g(442.0f, 0.0624f, 0.0374f) - 0.065f * g(501.1f, 0.0490f, 0.0382f));
            xyz[1] = std::max(0.0f, 0.821f * g(568.8f, 0.0213f, 0.0247f) + 0.286f * g(530.9f, 0.0613f, 0.0322f));
            xyz[2] = std::max(0.0f, 1.217f * g(437.0f, 0.0845f, 0.0278f) + 0.681f * g(459.0f, 0.0385f, 0.0725f));
            ySum += xyz[1];
        }
        for (auto& xyz : sceneData.cieXyz)
            for (int i = 0; i < 3; ++i) xyz[i] /= ySum;
        return sceneData;
    }

    void CreateSceneBuffers() {
        m_sceneDataBuffer = CreateBuffer(sizeof(SceneData), vk::BufferUsageFlagBits::eUniformBuffer);
        CreateMieScatteringBuffer();
        CreateRainbowScatteringBuffer();
    }

    void CreateMieScatteringBuffer() {
        const std::vector<MieMatrixEntry> table = ComputeMieScatteringTable(m_config.sky.spectral);
        const VkDeviceSize size = static_cast<VkDeviceSize>(table.size() * sizeof(MieMatrixEntry));
        m_mieScatteringBuffer = CreateBuffer(size, vk::BufferUsageFlagBits::eStorageBuffer);
        UploadToBuffer(m_mieScatteringBuffer, std::as_bytes(std::span{table}));
    }

    void CreateRainbowScatteringBuffer() {
        std::vector<MieMatrixEntry> table = ComputeRainbowScatteringTable(m_config.rainbow, m_config.sky.spectral.sunRadius);
        AppendRainbowSamplingCdf(table, int(m_config.rainbow.angleBins));
        const VkDeviceSize size = static_cast<VkDeviceSize>(table.size() * sizeof(MieMatrixEntry));
        m_rainbowScatteringBuffer = CreateBuffer(size, vk::BufferUsageFlagBits::eStorageBuffer);
        UploadToBuffer(m_rainbowScatteringBuffer, std::as_bytes(std::span{table}));
    }

    void UploadSceneDataFromConfig() {
        const SceneData sceneData = BuildSceneData();
        UploadToBuffer(m_sceneDataBuffer, std::as_bytes(std::span{&sceneData, 1}));
    }

    void CreateWindowAndShow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        m_window = glfwCreateWindow(static_cast<int>(m_config.render.width), static_cast<int>(m_config.render.height),
                                    "Vulkanic", nullptr, nullptr);
        glfwSetWindowUserPointer(m_window, this);
        glfwSetKeyCallback(m_window, KeyCallback);
        std::println("Controls: right-drag to look, R resets the view.");
        std::println("Polarizer: P toggles it, C switches linear/elliptical, [ ] adjusts it.");
    }

    vkb::Instance CreateInstance() {
        auto instanceResult = vkb::InstanceBuilder{}
                                  .set_app_name("Vulkan Path Tracer")
                                  .set_engine_name("None")
                                  .require_api_version(1, 4, 0)
                                  .enable_extension(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME)
                                  .enable_extension(VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME)
                                  .request_validation_layers()
                                  .build();

        m_instance = vk::raii::Instance(m_context, instanceResult.value().instance);
        return instanceResult.value();
    }

    void CreateSurface() {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        glfwCreateWindowSurface(static_cast<VkInstance>(*m_instance), m_window, nullptr, &surface);
        m_surface = vk::raii::SurfaceKHR(m_instance, surface);
    }

    vkb::PhysicalDevice PickPhysicalDevice(const vkb::Instance& vkbInstance) {
        VkPhysicalDeviceFeatures requiredFeatures{};
        requiredFeatures.shaderStorageImageWriteWithoutFormat = VK_TRUE;
        VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
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

        m_physicalDevice = vk::raii::PhysicalDevice(m_instance, deviceResult.value().physical_device);

        return deviceResult.value();
    }

    void CreateLogicalDevice(const vkb::PhysicalDevice& vkbPhysicalDevice) {
        auto deviceResult = vkb::DeviceBuilder{vkbPhysicalDevice}.build();
        m_device = vk::raii::Device(m_physicalDevice, deviceResult.value().device);

        m_queueFamily = deviceResult.value().get_queue_index(vkb::QueueType::graphics).value();
        m_graphicsQueue = m_device.getQueue(m_queueFamily, 0);
    }

    void CreateAllocator() {
        m_allocator = vma::raii::Allocator(m_instance, m_device,
                                           vma::AllocatorCreateInfo{}.setPhysicalDevice(*m_physicalDevice).setVulkanApiVersion(VK_API_VERSION_1_4));
    }

    vma::raii::Buffer CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage) const {
        return vma::raii::Buffer(m_allocator, vk::BufferCreateInfo{{}, size, usage},
                                 vma::AllocationCreateInfo{vma::AllocationCreateFlagBits::eHostAccessSequentialWrite, vma::MemoryUsage::eAuto});
    }

    static void UploadToBuffer(const vma::raii::Buffer& buffer, std::span<const std::byte> data) {
        buffer.getAllocation().copyFromMemory(data.data(), 0, data.size());
    }

    void CreateSceneResources() {
        CreateSceneBuffers();
        UploadSceneDataFromConfig();
    }

    void CreateSwapchain() {
        vkb::SwapchainBuilder builder{*m_physicalDevice, *m_device, *m_surface, m_queueFamily, m_queueFamily};
        builder.set_desired_format({VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
            .set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)
            .add_fallback_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
            .set_desired_extent(m_config.render.width, m_config.render.height)
            .set_image_usage_flags(VK_IMAGE_USAGE_STORAGE_BIT)
            .set_required_min_image_count(kFramesInFlight);
        auto swapchainResult = builder.build();

        vkb::Swapchain swapchain = swapchainResult.value();
        m_swapchain = vk::raii::SwapchainKHR(m_device, swapchain.swapchain);
        m_swapchainExtent = swapchain.extent;

        m_swapchainImages = m_swapchain.getImages();
        m_swapchainImageViews.clear();
        for (VkImageView view : swapchain.get_image_views().value()) {
            m_swapchainImageViews.emplace_back(m_device, view);
        }
    }

    void CreateDescriptorSetLayout() {
        using enum vk::DescriptorType;
        constexpr auto compute = vk::ShaderStageFlagBits::eCompute;
        const std::array<vk::DescriptorSetLayoutBinding, 4> layoutBindings{{
            {0, eStorageImage, 1, compute},
            {2, eUniformBuffer, 1, compute},
            {7, eStorageBuffer, 1, compute},
            {8, eStorageBuffer, 1, compute},
        }};
        vk::DescriptorSetLayoutCreateInfo createInfo{vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor};
        createInfo.setBindings(layoutBindings);
        m_descriptorSetLayout = vk::raii::DescriptorSetLayout(m_device, createInfo);
    }

    vk::raii::ShaderModule CreateShaderModule(std::span<const uint32_t> spirv) {
        return vk::raii::ShaderModule(m_device, vk::ShaderModuleCreateInfo{{}, spirv.size_bytes(), spirv.data()});
    }

    void CreatePipeline() {
        const vk::raii::ShaderModule computeModule = CreateShaderModule(kPathTracerSpirv);

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

        const vk::PipelineShaderStageCreateInfo stage{{}, vk::ShaderStageFlagBits::eCompute, *computeModule, "main", &specialization};
        m_computePipeline = vk::raii::Pipeline(m_device, nullptr, vk::ComputePipelineCreateInfo{{}, stage, *m_pipelineLayout});
    }

    void CreateCommandPool() {
        vk::CommandPoolCreateInfo createInfo{vk::CommandPoolCreateFlagBits::eResetCommandBuffer, m_queueFamily};
        m_commandPool = vk::raii::CommandPool(m_device, createInfo);
    }

    void CreateFrameResources() {
        const uint32_t count = kFramesInFlight;
        vk::raii::CommandBuffers commandBuffers(m_device, {*m_commandPool, vk::CommandBufferLevel::ePrimary, count});
        const vk::FenceCreateInfo signaled{vk::FenceCreateFlagBits::eSignaled};
        m_frames.clear();
        for (uint32_t i = 0; i < count; ++i) {
            m_frames.push_back({std::move(commandBuffers[i]), m_device.createSemaphore({}), m_device.createSemaphore({}),
                                m_device.createFence(signaled), m_device.createFence(signaled)});
        }
    }

    PushConstants BuildPushConstants() {
        const CameraController::View view = m_camera.GetView();

        const float tanHalfFov = std::tan(m_config.camera.fovYDegrees * kPi / 360.0f);
        const float aspect = float(m_swapchainExtent.width) / float(m_swapchainExtent.height);
        const float fx = std::sin(view.yaw) * std::cos(view.pitch), fy = std::sin(view.pitch), fz = std::cos(view.yaw) * std::cos(view.pitch);
        const float rx = std::cos(view.yaw), rz = -std::sin(view.yaw);
        const PushConstants constants{
            .forward = {fx, fy, fz, float(m_config.render.samplesPerPixel)},
            .right = {rx * aspect * tanHalfFov, 0.0f, rz * aspect * tanHalfFov, 0.0f},
            .up = {fy * rz * tanHalfFov, (fz * rx - fx * rz) * tanHalfFov, -fy * rx * tanHalfFov, 0.0f},
            .frame = {float(m_frameIndex), m_config.sky.exposure, 0.0f, 0.0f},
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
        imageBarrier(Stage::eNone, Access::eNone, Stage::eComputeShader, Access::eShaderStorageWrite,
                     vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *m_computePipeline);
        using enum vk::DescriptorType;
        const vk::DescriptorImageInfo imageInfo{{}, *m_swapchainImageViews[imageIndex], vk::ImageLayout::eGeneral};
        const auto bufferInfo = [](const vma::raii::Buffer& b) { return vk::DescriptorBufferInfo{*b, 0, vk::WholeSize}; };
        const std::array bufferInfos{bufferInfo(m_sceneDataBuffer), bufferInfo(m_mieScatteringBuffer),
                                     bufferInfo(m_rainbowScatteringBuffer)};
        const std::array writes{
            vk::WriteDescriptorSet{{}, 0, 0, eStorageImage, imageInfo},
            vk::WriteDescriptorSet{{}, 2, 0, eUniformBuffer, {}, bufferInfos[0]},
            vk::WriteDescriptorSet{{}, 7, 0, eStorageBuffer, {}, bufferInfos[1]},
            vk::WriteDescriptorSet{{}, 8, 0, eStorageBuffer, {}, bufferInfos[2]},
        };
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
        const std::array fences{*frame.inFlight, *frame.presentDone};
        while (m_device.waitForFences(fences, VK_TRUE, UINT64_MAX) == vk::Result::eTimeout) {
        }

        const uint32_t imageIndex = m_swapchain.acquireNextImage(UINT64_MAX, *frame.imageAvailable).value;

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
        std::ignore = m_graphicsQueue.presentKHR(presentInfo);

        ++m_frameIndex;
        m_currentFrame = (m_currentFrame + 1) % static_cast<uint32_t>(m_frames.size());
    }

    void MessageLoop() {
        using Clock = std::chrono::steady_clock;
        auto previousFrame = Clock::now();
        auto titleUpdate = previousFrame;
        uint32_t frames = 0;

        while (!glfwWindowShouldClose(m_window)) {
            glfwPollEvents();

            const auto now = Clock::now();
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
        vk::raii::Semaphore renderFinished{nullptr};
        vk::raii::Fence presentDone{nullptr};
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
    vma::raii::Buffer m_sceneDataBuffer{nullptr};
    vma::raii::Buffer m_mieScatteringBuffer{nullptr};
    vma::raii::Buffer m_rainbowScatteringBuffer{nullptr};

    vk::raii::SwapchainKHR m_swapchain{nullptr};
    vk::Extent2D m_swapchainExtent{};
    std::vector<vk::Image> m_swapchainImages;
    std::vector<vk::raii::ImageView> m_swapchainImageViews;

    vk::raii::DescriptorSetLayout m_descriptorSetLayout{nullptr};
    vk::raii::PipelineLayout m_pipelineLayout{nullptr};
    vk::raii::Pipeline m_computePipeline{nullptr};

    vk::raii::CommandPool m_commandPool{nullptr};
    std::vector<FrameResources> m_frames;
    uint32_t m_currentFrame = 0;
    uint64_t m_frameIndex = 0;
    RuntimeConfig m_config{};

    CameraController m_camera;
};

int main() {
    VulkanPathTracer app;
    app.Run();
}
