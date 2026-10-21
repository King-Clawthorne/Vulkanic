/// @file App.cpp
/// @brief Creates the Vulkan window and runs the interactive sky renderer.
///
/// This file connects runtime settings to the
/// compute shader, manages per-frame Vulkan resources, and translates camera
/// and polarizer input into shader parameters.
#include <VkBootstrap.h>
#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#ifndef __clang_analyzer__
#define VMA_IMPLEMENTATION
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <format>
#include <glm/glm.hpp>
#include <numbers>
#include <optional>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vk_mem_alloc_raii.hpp>

/// Single-precision pi used by the renderer and its camera calculations.
inline constexpr float kPi = std::numbers::pi_v<float>;

/// Parameters used to build atmospheric shader constants.
/// Sun direction is in world space. Radii and scale heights are in metres.
struct SkyConfig {
	float betaRayleighGreen = 13.5e-6f; ///< Green-channel Rayleigh scattering coefficient at 550 nm, in inverse metres.
	float earthRadius = 6360e3f; ///< Planet radius, in metres.
	float atmosphereRadius = 6420e3f; ///< Outer radius of the modeled atmosphere, in metres.
	float scaleHeightRayleigh = 7994.0f; ///< Rayleigh density scale height, in metres.
	float sunRadiance = 8317.742f; ///< Equal solar radiance per color channel, in renderer radiance units.
	glm::vec3 sunDirection { 0.35f, 0.01f, 0.25f }; ///< Direction toward the Sun in world space.
	float sunRadius = 0.00465f; ///< Angular radius of the Sun, in radians.
	float rayleighDepolarization = 0.0279f; ///< Depolarization parameter used by the Rayleigh Mueller matrix.
	/// Compares all sky parameters for exact equality.
	[[nodiscard]] friend bool operator==(const SkyConfig&, const SkyConfig&) = default;
};

/// Output image width and height, in pixels, used for the window and swapchain.
struct RenderConfig {
	glm::uvec2 resolution = { 1280, 720 }; ///< Requested swapchain width and height, in pixels.
};

/// Initial camera angles and vertical field-of-view limits.
/// Yaw and pitch are in radians. Fields suffixed with Degrees use degrees.
struct CameraConfig {
	glm::vec2 initialAngles { 0.951f, 0.0f }; ///< Initial camera yaw and pitch, in radians.
	float fovYDegrees = 40.0f; ///< Vertical camera field of view, in degrees.
	float maxPitchDegrees = 89.0f; ///< Maximum absolute camera pitch, in degrees.
	float exposure = 10.0f; ///< Display exposure multiplier applied to integrated radiance.
};

/// Mouse-look sensitivity in radians per cursor pixel and polarizer rotation rate in radians per second.
struct InputConfig {
	float mouseSensitivity = 0.0035f; ///< Camera rotation in radians per cursor pixel.
	float polarizerRotateSpeed = 1.5f; ///< Polarizer rotation in radians per second.
};

/// Complete set of renderer settings consumed by the application.
struct RuntimeConfig {
	RenderConfig render { }; ///< Window and swapchain dimensions.
	CameraConfig camera { }; ///< Initial camera pose, field of view, and exposure.
	InputConfig input { }; ///< Mouse and polarizer control rates.
	SkyConfig sky { }; ///< Atmosphere, Sun, and scattering parameters.
};

/// Approximate Rayleigh scattering strength for red, green, and blue relative
/// to green, using an inverse fourth-power fit at 650, 550, and 450 nm.
inline constexpr glm::vec3 kRayleighScale = { 0.512f, 1.0f, 2.231f };

/// Tracks the view and polarizer controls used by the renderer.
struct CameraController {
	/// Render-affecting camera and polarizer state for one frame.
	struct View {
		glm::vec2 angles; ///< Camera yaw and pitch, in radians.
		bool polarizerEnabled; ///< Whether the polarizer is applied.
		float polarizerAngle; ///< Linear polarizer angle, in radians.
		float polarizerEllipticity; ///< Ellipticity angle, in radians, zero for linear polarization.
		/// Compares camera and polarizer states.
		friend bool operator==(const View&, const View&) = default;
	};

	glm::vec2 angles = { 0.0f, 0.0f }; ///< Current camera yaw and pitch, in radians.
	float polarizerAngle = 0.0f; ///< Current linear polarizer angle, in radians.
	float ellipticity = kPi * 0.125f; ///< Current ellipticity angle, in radians.
	double lastMouseX = 0.0; ///< Cursor X position from the previous update.
	double lastMouseY = 0.0; ///< Cursor Y position from the previous update.
	bool mouseLookActive = false; ///< Whether right-drag camera look was active last update.
	bool polarizerEnabled = false; ///< Whether the polarizer is enabled.
	bool elliptical = false; ///< Whether polarizer rotation edits ellipticity rather than angle.

	/// Initializes yaw and pitch from the camera configuration.
	/// @param config Runtime camera limits and input sensitivities.
	explicit CameraController(const RuntimeConfig& config) {
		angles = config.camera.initialAngles;
	}

	/// Applies one-shot keyboard controls for the polarizer modes.
	/// @param key GLFW key code that was pressed.
	void OnKeyPress(int key) {
		if (key == GLFW_KEY_P) polarizerEnabled = !polarizerEnabled;
		if (key == GLFW_KEY_C) elliptical = !elliptical;
	}

	/// Updates camera look and polarizer rotation from the current window input.
	/// @param deltaSeconds Time since the preceding update, in seconds.
	/// @param window GLFW window from which focus, cursor, and key state are read.
	/// @param config Runtime input and camera limits.
	void Update(double deltaSeconds, GLFWwindow* window, const RuntimeConfig& config) {
		if (glfwGetWindowAttrib(window, GLFW_FOCUSED) != GLFW_TRUE) {
			mouseLookActive = false;
			return;
		}

		const float dt = static_cast<float>(std::min(deltaSeconds, 0.1));

		double x = 0.0, y = 0.0;
		glfwGetCursorPos(window, &x, &y);
		const bool mouseLook = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

		if (mouseLook && mouseLookActive) {
			angles.x += static_cast<float>(x - lastMouseX) * config.input.mouseSensitivity;
			angles.y -= static_cast<float>(y - lastMouseY) * config.input.mouseSensitivity;
		}

		mouseLookActive = mouseLook;
		lastMouseX = x;
		lastMouseY = y;

		const float maxPitch = config.camera.maxPitchDegrees * kPi / 180.0f;
		angles.y = std::clamp(angles.y, -maxPitch, maxPitch);

		const float axis = static_cast<float>(glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) - static_cast<float>(glfwGetKey(window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS);
		const float rotation = axis * config.input.polarizerRotateSpeed * dt;

		if (elliptical) ellipticity = std::clamp(ellipticity + rotation, -kPi * 0.25f, kPi * 0.25f);
		else polarizerAngle += rotation;
	}

	/// Returns a snapshot of the state that affects rendered pixels.
	/// @return Current camera angles and polarizer state.
	[[nodiscard]] View GetView() const { return { .angles = angles, .polarizerEnabled = polarizerEnabled, .polarizerAngle = polarizerAngle, .polarizerEllipticity = elliptical ? ellipticity : 0.0f }; }
};

/// Number of command buffers and synchronization slots kept in flight.
constexpr uint32_t kFramesInFlight = 2;
/// Compute shader SPIR-V embedded by the build from Main.comp.
constexpr uint32_t kPathTracerSpirv[] = {
#include "Main.comp.spv.inc"
};

/// Uniform-buffer payload shared with the compute shader.
///
/// Members are packed into 16-byte vectors to match the shader's std140 layout.
/// The static assertions below guard the offsets and total size.
struct alignas(16) SceneData {
	glm::vec4 skyParams; ///< Earth radius, atmosphere radius, Rayleigh scale height, and depolarization.
	glm::vec4 skySunDirectionRadius; ///< Sun direction in xyz and angular radius in radians.
	glm::vec4 rayleigh; ///< Per-channel Rayleigh scattering coefficients.
	glm::vec4 sunParams; ///< Solar disk solid angle, equal-channel radiance, and padding.
};

static_assert(offsetof(SceneData, skySunDirectionRadius) == 16);
static_assert(offsetof(SceneData, sunParams) == 48);
static_assert(offsetof(SceneData, rayleigh) == 32);
static_assert(sizeof(SceneData) == 64);

/// Descriptor types in the binding order expected by Main.comp.
constexpr std::array kDescriptorTypes { vk::DescriptorType::eStorageImage, vk::DescriptorType::eUniformBuffer };

/// Per-dispatch camera, polarizer, exposure, and image parameters.
struct PushConstants {
	glm::vec4 forward; ///< Camera forward direction.
	glm::vec4 right; ///< Horizontal camera basis scaled by aspect ratio and field of view.
	glm::vec4 up; ///< Vertical camera basis scaled by field of view.
	glm::vec4 polarizer; ///< Enabled flag, linear angle, ellipticity, and exposure.
	glm::uvec2 imageSize; ///< Swapchain width and height in pixels.
};

/// Owns the window, Vulkan objects, scene buffers, and render loop.
class App {
public:
	/// Creates the application state and initializes the camera from default settings.
	App()
		: m_camera(m_config) { }

	/// Waits for outstanding device work and releases the GLFW window.
	~App() {
		if (*m_device) vkDeviceWaitIdle(*m_device);
		if (m_window != nullptr) {
			glfwDestroyWindow(m_window);
			m_window = nullptr;
		}

		glfwTerminate();
	}

	/// Initializes the application and runs until the window closes.
	/// @return Zero on a normal exit. Nonzero if window or Vulkan setup fails.
	int Run() {
		if (!CreateWindowAndShow()) return 1;
		if (!CreateVulkan()) return 1;

		m_allocator = vma::raii::Allocator(m_instance, m_device, vma::AllocatorCreateInfo { }.setPhysicalDevice(*m_physicalDevice).setVulkanApiVersion(VK_API_VERSION_1_4));
		m_commandPool = vk::raii::CommandPool(m_device, { vk::CommandPoolCreateFlagBits::eResetCommandBuffer, m_queueFamily });

		CreateSceneBuffers();
		CreateSwapchain();
		CreateDescriptorSetLayout();
		CreatePipeline();
		CreateFrameResources();
		MessageLoop();

		m_device.waitIdle();
		return 0;
	}

private:
	/// Forwards GLFW key presses to the camera controller stored on the window.
	/// @param window GLFW window that received the event.
	/// @param key GLFW key code.
	/// @param scancode Platform-specific key code, unused here.
	/// @param action GLFW key action. Only presses are handled.
	/// @param mods Active GLFW modifier flags, unused here.
	static void KeyCallback(GLFWwindow* window, int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods) {
		auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));
		if (app == nullptr || action != GLFW_PRESS) return;
		app->m_camera.OnKeyPress(key);
	}

	/// Packs the configured values for the shader.
	/// @return Uniform-buffer payload matching the shader's SceneData layout.
	[[nodiscard]] SceneData BuildSceneData() const {
		const SkyConfig& s = m_config.sky;
		SceneData sceneData {
			.skyParams = { s.earthRadius, s.atmosphereRadius, s.scaleHeightRayleigh, s.rayleighDepolarization },
			.skySunDirectionRadius = { s.sunDirection[0], s.sunDirection[1], s.sunDirection[2], s.sunRadius },
			.rayleigh = { s.betaRayleighGreen * kRayleighScale[0], s.betaRayleighGreen * kRayleighScale[1], s.betaRayleighGreen * kRayleighScale[2], 0.0f },
			.sunParams = { 2.0f * kPi * (1.0f - std::cos(s.sunRadius)), s.sunRadiance, 0.0f, 0.0f },
		};

		return sceneData;
	}

	/// Allocates and uploads the scene uniform data.
	void CreateSceneBuffers() {
		m_sceneDataBuffer = CreateBuffer(sizeof(SceneData), vk::BufferUsageFlagBits::eUniformBuffer);
		const SceneData sceneData = BuildSceneData();
		UploadToBuffer(m_sceneDataBuffer, std::as_bytes(std::span { &sceneData, 1 }));
	}

	/// Initializes GLFW, creates the fixed-size window, and installs input callbacks.
	/// @return Whether GLFW and window creation succeeded.
	bool CreateWindowAndShow() {
		if (glfwInit() != GLFW_TRUE) {
			const char* description = nullptr;
			const int error = glfwGetError(&description);
			std::println(stderr, "GLFW initialization failed ({}): {}", error, description != nullptr ? description : "no details");
			return false;
		}

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		m_window = glfwCreateWindow(static_cast<int>(m_config.render.resolution.x), static_cast<int>(m_config.render.resolution.y), "Vulkanic", nullptr, nullptr);
		if (m_window == nullptr) {
			const char* description = nullptr;
			const int error = glfwGetError(&description);
			std::println(stderr, "GLFW window creation failed ({}): {}", error, description != nullptr ? description : "no details");
			glfwTerminate();
			return false;
		}

		glfwSetWindowUserPointer(m_window, this);
		glfwSetKeyCallback(m_window, KeyCallback);
		return true;
	}

	/// Creates the Vulkan instance, surface, physical device, logical device, and queues.
	/// @return Whether surface creation succeeded. Unsupported Vulkan requirements throw.
	bool CreateVulkan() {
		m_selectedInstance = vkb::InstanceBuilder { }
								 .set_app_name("Vulkan Path Tracer")
								 .set_engine_name("Vulkanic")
								 .require_api_version(1, 4)
								 .enable_extension(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME)
								 .enable_extension(VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME)
								 .build()
								 .value();

		m_instance = vk::raii::Instance(m_context, m_selectedInstance.instance);
		VkSurfaceKHR surface = VK_NULL_HANDLE;
		const VkResult surfaceResult = glfwCreateWindowSurface(static_cast<VkInstance>(*m_instance), m_window, nullptr, &surface);

		if (surfaceResult != VK_SUCCESS) {
			std::println(stderr, "Vulkan surface creation failed (VkResult {}).", static_cast<int>(surfaceResult));
			return false;
		}

		m_surface = vk::raii::SurfaceKHR(m_instance, surface);
		VkPhysicalDeviceFeatures requiredFeatures { };
		requiredFeatures.shaderStorageImageWriteWithoutFormat = VK_TRUE;
		VkPhysicalDeviceVulkan13Features requiredFeatures13 { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
		requiredFeatures13.synchronization2 = VK_TRUE;
		VkPhysicalDeviceVulkan14Features requiredFeatures14 { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES };
		requiredFeatures14.pushDescriptor = VK_TRUE;

		m_selectedPhysicalDevice = vkb::PhysicalDeviceSelector { m_selectedInstance }
									   .set_minimum_version(1, 4)
									   .set_surface(static_cast<VkSurfaceKHR>(*m_surface))
									   .add_required_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME)
									   .set_required_features(requiredFeatures)
									   .set_required_features_13(requiredFeatures13)
									   .set_required_features_14(requiredFeatures14)
									   .select()
									   .value();

		m_physicalDevice = vk::raii::PhysicalDevice(m_instance, m_selectedPhysicalDevice.physical_device);
		m_selectedDevice = vkb::DeviceBuilder { m_selectedPhysicalDevice }.build().value();
		const auto graphicsQueueFamily = m_selectedDevice.get_queue_index(vkb::QueueType::graphics);
		const auto presentQueueFamily = m_selectedDevice.get_queue_index(vkb::QueueType::present);
		m_queueFamily = graphicsQueueFamily.value();
		m_device = vk::raii::Device(m_physicalDevice, m_selectedDevice.device);
		m_graphicsQueue = m_device.getQueue(m_queueFamily, 0);
		m_presentQueue = m_device.getQueue(presentQueueFamily.value(), 0);
		return true;
	}

	/// Allocates a VMA buffer with the requested Vulkan usage and memory flags.
	/// @param size Buffer size in bytes.
	/// @param usage Vulkan buffer usage flags.
	/// @param flags VMA allocation flags. Defaults to sequential host writes.
	/// @return The allocated buffer.
	[[nodiscard]] vma::raii::Buffer
	CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vma::AllocationCreateFlags flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite) const {
		return vma::raii::Buffer(m_allocator, vk::BufferCreateInfo { { }, size, usage }, vma::AllocationCreateInfo { flags, vma::MemoryUsage::eAuto });
	}

	/// Copies host bytes into a VMA buffer allocation.
	/// @param buffer Destination buffer.
	/// @param data Bytes to copy, starting at offset zero.
	static void UploadToBuffer(const vma::raii::Buffer& buffer, std::span<const std::byte> data) { buffer.getAllocation().copyFromMemory(data.data(), 0, data.size()); }

	/// Creates the presentation swapchain, image views, and render-finished semaphores.
	void CreateSwapchain() {
		auto swapchainResult = vkb::SwapchainBuilder { m_selectedDevice }
								   .set_desired_format({ .format = VK_FORMAT_B8G8R8A8_UNORM, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
								   .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
								   .set_image_usage_flags(VK_IMAGE_USAGE_STORAGE_BIT)
								   .set_desired_extent(m_config.render.resolution.x, m_config.render.resolution.y)
								   .set_desired_min_image_count(kFramesInFlight)
								   .build();

		m_selectedSwapchain = swapchainResult.value();
		m_swapchainExtent = vk::Extent2D { m_selectedSwapchain.extent.width, m_selectedSwapchain.extent.height };
		m_swapchain = vk::raii::SwapchainKHR(m_device, m_selectedSwapchain.swapchain);
		auto imagesAndViews = m_selectedSwapchain.get_images_and_image_views();
		const auto& [images, imageViews] = imagesAndViews.value();
		m_swapchainImages.assign(images.begin(), images.end());
		m_swapchainImageViews.clear();
		m_renderFinished.clear();

		for (size_t i = 0; i < images.size(); i++) {
			m_swapchainImageViews.emplace_back(m_device, imageViews[i]);
			m_renderFinished.emplace_back(m_device.createSemaphore({ }));
		}

		m_selectedSwapchain.swapchain = VK_NULL_HANDLE;
	}

	/// Creates the push-descriptor layout used by the compute shader.
	void CreateDescriptorSetLayout() {
		std::array<vk::DescriptorSetLayoutBinding, kDescriptorTypes.size()> layoutBindings { };
		for (const auto [binding, type] : std::views::enumerate(kDescriptorTypes)) layoutBindings[binding] = { static_cast<uint32_t>(binding), type, 1, vk::ShaderStageFlagBits::eCompute };
		vk::DescriptorSetLayoutCreateInfo createInfo { vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor };
		createInfo.setBindings(layoutBindings);
		m_descriptorSetLayout = vk::raii::DescriptorSetLayout(m_device, createInfo);
	}

	/// Creates the compute pipeline with its descriptor and push-constant layouts.
	void CreatePipeline() {
		const vk::raii::ShaderModule computeModule { m_device, vk::ShaderModuleCreateInfo { { }, sizeof(kPathTracerSpirv), std::data(kPathTracerSpirv) } };
		const vk::PushConstantRange pushRange { vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants) };
		vk::PipelineLayoutCreateInfo layoutInfo { };
		layoutInfo.setSetLayouts(*m_descriptorSetLayout);
		layoutInfo.setPushConstantRanges(pushRange);
		m_pipelineLayout = vk::raii::PipelineLayout(m_device, layoutInfo);
		const vk::PipelineShaderStageCreateInfo stage { { }, vk::ShaderStageFlagBits::eCompute, *computeModule, "main", nullptr };
		m_computePipeline = vk::raii::Pipeline(m_device, nullptr, vk::ComputePipelineCreateInfo { { }, stage, *m_pipelineLayout });
	}

	/// Creates per-frame command buffers and synchronization.
	void CreateFrameResources() {
		const uint32_t count = kFramesInFlight;
		vk::raii::CommandBuffers commandBuffers(m_device, { *m_commandPool, vk::CommandBufferLevel::ePrimary, count });
		const vk::FenceCreateInfo signaled { vk::FenceCreateFlagBits::eSignaled };
		m_frames.clear();

		for (uint32_t i = 0; i < count; i++) m_frames.push_back({ .commandBuffer = std::move(commandBuffers[i]), .imageAvailable = m_device.createSemaphore({ }), .inFlight = m_device.createFence(signaled) });
	}

	/// Builds shader parameters from the current view.
	/// @return Push-constant payload for the next compute dispatch.
	PushConstants BuildPushConstants() {
		const CameraController::View view = m_camera.GetView();

		const float tanHalfFov = std::tan(m_config.camera.fovYDegrees * kPi / 360.0f);
		const float aspect = static_cast<float>(m_swapchainExtent.width) / static_cast<float>(m_swapchainExtent.height);
		const float fx = std::sin(view.angles.x) * std::cos(view.angles.y);
		const float fy = std::sin(view.angles.y);
		const float fz = std::cos(view.angles.x) * std::cos(view.angles.y);
		const float rx = std::cos(view.angles.x);
		const float rz = -std::sin(view.angles.x);

		const PushConstants constants {
			.forward = { fx, fy, fz, 0.0f },
			.right = { rx * aspect * tanHalfFov, 0.0f, rz * aspect * tanHalfFov, 0.0 },
			.up = { fy * rz * tanHalfFov, ((fz * rx) - (fx * rz)) * tanHalfFov, -fy * rx * tanHalfFov, 0.0 },
			.polarizer = { view.polarizerEnabled ? 1.0f : 0.0f, view.polarizerAngle, view.polarizerEllipticity, m_config.camera.exposure },
			.imageSize = { m_swapchainExtent.width, m_swapchainExtent.height },
		};

		return constants;
	}

	/// Records barriers, descriptors, compute dispatch, and presentation transition.
	/// @param commandBuffer Command buffer to record.
	/// @param imageIndex Acquired swapchain image index for this frame.
	void RecordCommandBuffer(const vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex) {
		using Stage = vk::PipelineStageFlagBits2;
		using Access = vk::AccessFlagBits2;
		const vk::Image image = m_swapchainImages[imageIndex];
		const vk::ImageSubresourceRange colorRange { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
		const auto imageBarrier = [&](vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess, vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
			const vk::ImageMemoryBarrier2 barrier { srcStage, srcAccess, dstStage, dstAccess, oldLayout, newLayout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, colorRange };
			commandBuffer.pipelineBarrier2(vk::DependencyInfo { }.setImageMemoryBarriers(barrier));
		};

		commandBuffer.begin({ });
		imageBarrier(Stage::eNone, Access::eNone, Stage::eComputeShader, Access::eShaderStorageWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

		commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *m_computePipeline);
		const vk::DescriptorImageInfo imageInfo { { }, *m_swapchainImageViews[imageIndex], vk::ImageLayout::eGeneral };
		const auto bufferInfo = [](const vma::raii::Buffer& b) { return vk::DescriptorBufferInfo { *b, 0, vk::WholeSize }; };
		const std::array bufferInfos { bufferInfo(m_sceneDataBuffer) };

		std::array<vk::WriteDescriptorSet, kDescriptorTypes.size()> writes { };
		writes[0] = vk::WriteDescriptorSet { { }, 0, 0, kDescriptorTypes[0], imageInfo };
		for (uint32_t i = 1; i < writes.size(); i++) writes[i] = vk::WriteDescriptorSet { { }, i, 0, kDescriptorTypes[i], { }, bufferInfos[i - 1] };

		commandBuffer.pushDescriptorSet(vk::PipelineBindPoint::eCompute, *m_pipelineLayout, 0, writes);
		commandBuffer.pushConstants<PushConstants>(*m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, BuildPushConstants());

		constexpr uint32_t kTile = 8;
		const uint32_t groupsX = (m_swapchainExtent.width + kTile - 1) / kTile;
		const uint32_t groupsY = (m_swapchainExtent.height + kTile - 1) / kTile;
		commandBuffer.dispatch(groupsX, groupsY, 1);

		imageBarrier(Stage::eComputeShader, Access::eShaderStorageWrite, Stage::eNone, Access::eNone, vk::ImageLayout::eGeneral, vk::ImageLayout::ePresentSrcKHR);
		commandBuffer.end();
	}

	/// Acquires an image, submits its compute work, and presents the result.
	void RenderFrame() {
		FrameResources& frame = m_frames[m_currentFrame];
		while (m_device.waitForFences(*frame.inFlight, VK_TRUE, UINT64_MAX) == vk::Result::eTimeout) { }
		const uint32_t imageIndex = m_swapchain.acquireNextImage(UINT64_MAX, *frame.imageAvailable).value;

		m_device.resetFences(*frame.inFlight);
		const vk::raii::CommandBuffer& commandBuffer = frame.commandBuffer;
		commandBuffer.reset();
		RecordCommandBuffer(commandBuffer, imageIndex);

		const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eComputeShader;
		vk::SubmitInfo submitInfo { };
		submitInfo.setWaitSemaphores(*frame.imageAvailable);
		submitInfo.setWaitDstStageMask(waitStage);
		submitInfo.setCommandBuffers(*commandBuffer);
		submitInfo.setSignalSemaphores(*m_renderFinished[imageIndex]);
		m_graphicsQueue.submit(submitInfo, *frame.inFlight);

		vk::PresentInfoKHR presentInfo { };
		presentInfo.setWaitSemaphores(*m_renderFinished[imageIndex]);
		presentInfo.setSwapchains(*m_swapchain);
		presentInfo.setImageIndices(imageIndex);
		std::ignore = m_presentQueue.presentKHR(presentInfo);

		m_currentFrame = (m_currentFrame + 1) % static_cast<uint32_t>(m_frames.size());
	}

	/// Polls window input, updates the camera, renders frames, and reports frame rate.
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

			frames++;
			const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - titleUpdate).count();
			if (elapsed >= 1.0) {
				m_displayFps = static_cast<float>(frames / elapsed);
				const std::string title = std::format("Vulkanic - {:.1f} FPS ({:.2f} ms)", m_displayFps, 1000.0 * elapsed / frames);
				glfwSetWindowTitle(m_window, title.c_str());
				frames = 0;
				titleUpdate = std::chrono::steady_clock::now();
			}
		}
	}

	/// Command and synchronization objects associated with one frame slot.
	struct FrameResources {
		vk::raii::CommandBuffer commandBuffer { nullptr }; ///< Commands recorded for this frame slot.
		vk::raii::Semaphore imageAvailable { nullptr }; ///< Signals that a swapchain image can be rendered.
		vk::raii::Fence inFlight { nullptr }; ///< Signals completion of the submitted frame.
	};

	GLFWwindow* m_window = nullptr; ///< GLFW window used for input, presentation, and event polling.

	vk::raii::Context m_context; ///< Vulkan-Hpp context used to create RAII handles.
	vkb::Instance m_selectedInstance { }; ///< VkBootstrap instance result used to initialize m_instance.
	vkb::PhysicalDevice m_selectedPhysicalDevice { }; ///< VkBootstrap physical-device selection and queue details.
	vkb::Device m_selectedDevice { }; ///< VkBootstrap logical-device selection used to create queues and the swapchain.
	vkb::Swapchain m_selectedSwapchain { }; ///< VkBootstrap swapchain metadata while importing its handles.
	vk::raii::Instance m_instance { nullptr }; ///< RAII handle for the Vulkan instance.
	vk::raii::SurfaceKHR m_surface { nullptr }; ///< Vulkan presentation surface for m_window.
	vk::raii::PhysicalDevice m_physicalDevice { nullptr }; ///< RAII handle for the selected physical device.
	vk::raii::Device m_device { nullptr }; ///< RAII handle for the logical device.
	uint32_t m_queueFamily = 0; ///< Queue family used to allocate and submit compute command buffers.
	vk::raii::Queue m_graphicsQueue { nullptr }; ///< Queue used for compute submission and rendering work.
	vk::raii::Queue m_presentQueue { nullptr }; ///< Queue used to present completed swapchain images.

	vma::raii::Allocator m_allocator { nullptr }; ///< VMA allocator used for renderer buffers.
	vma::raii::Buffer m_sceneDataBuffer { nullptr }; ///< Uniform data shared with the compute shader.

	vk::raii::SwapchainKHR m_swapchain { nullptr }; ///< RAII handle for the presentation swapchain.
	vk::Extent2D m_swapchainExtent { }; ///< Actual swapchain image dimensions, in pixels.
	std::vector<vk::Image> m_swapchainImages; ///< Swapchain images used as compute outputs.
	std::vector<vk::raii::ImageView> m_swapchainImageViews; ///< Storage-image views for the swapchain images.
	std::vector<vk::raii::Semaphore> m_renderFinished; ///< Signals completion for each swapchain image.

	vk::raii::DescriptorSetLayout m_descriptorSetLayout { nullptr }; ///< Push-descriptor bindings consumed by Main.comp.
	vk::raii::PipelineLayout m_pipelineLayout { nullptr }; ///< Descriptor and push-constant layout for the compute pipeline.
	vk::raii::Pipeline m_computePipeline { nullptr }; ///< Vulkan compute pipeline for the embedded path-tracing shader.
	vk::raii::CommandPool m_commandPool { nullptr }; ///< Command pool for per-frame compute command buffers.

	std::vector<FrameResources> m_frames; ///< Command and synchronization resources for frames in flight.
	uint32_t m_currentFrame = 0; ///< Index of the frame resource used by the next submission.
	float m_displayFps = 0.0f; ///< Most recently measured frame rate shown in the window title.
	RuntimeConfig m_config { }; ///< Renderer configuration used to build scene and dispatch parameters.
	CameraController m_camera; ///< Interactive camera and polarizer state.
};

/// Application entry point. Reports setup and runtime exceptions to stderr.
/// @return Zero after normal shutdown, or nonzero when initialization or runtime fails.
int main() {
	try {
		App app;
		return app.Run();
	} catch (const std::exception& error) {
		std::fputs("Vulkanic failed: ", stderr);
		std::fputs(error.what(), stderr);
		std::fputc('\n', stderr);
		return 1;
	} catch (...) {
		std::fputs("Vulkanic failed with an unknown exception.\n", stderr);
		return 1;
	}
}
