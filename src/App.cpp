/// @file App.cpp
/// @brief Creates the Vulkan window and runs the interactive spectral path tracer.
///
/// This file connects runtime settings and precomputed scattering tables to the
/// compute shader, manages per-frame Vulkan resources, and translates camera
/// and polarizer input into shader parameters.
#include "Settings.hpp"
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
#include <iterator>
#include <optional>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vk_mem_alloc_raii.hpp>

/// Tracks the view and polarizer controls used by the renderer.
struct CameraController {
	/// Render-affecting camera and polarizer state for one frame.
	struct View {
		glm::vec2 angles; ///< Camera yaw and pitch, in radians.
		bool polarizerEnabled; ///< Whether the polarizer is applied.
		float polarizerAngle; ///< Linear polarizer angle, in radians.
		float polarizerEllipticity; ///< Ellipticity angle, in radians; zero for linear polarization.
		/// Compares the state that determines whether accumulated samples remain valid.
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

	explicit CameraController(const RuntimeConfig& config) {
		const glm::vec3 delta = config.camera.initialLookAt - config.camera.initialPosition;
		angles.x = std::atan2(delta.x, delta.z);
		angles.y = std::atan2(delta.y, std::hypot(delta.x, delta.z));
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
/// the static assertions below guard the offsets and total size.
struct alignas(16) SceneData {
	glm::vec4 skySpectralParams; ///< 550 nm Rayleigh coefficient, first aerosol beta, reserved, and sun radiance.
	glm::vec4 skyRadiiScaleHeights; ///< Earth radius, atmosphere radius, Rayleigh scale height, and first aerosol scale height.
	glm::vec4 skySunDirectionRadius; ///< Sun direction in xyz and angular radius in radians.
	glm::vec4 skyVrtParams; ///< Sun antialiasing angle, Rayleigh depolarization, Mie angle-bin count, and second aerosol scale height.
	glm::vec4 rainbowCenterEnabled; ///< Rainbow center in xyz and enabled flag in w.
	glm::vec4 rainbowRadiiEdge; ///< Configured rainbow radii and edge softness.
	glm::vec4 rainbowOptical; ///< Rainbow scattering/extinction coefficients, angle-bin count, and view-step count.
	glm::uvec4 rainbowMultiple; ///< Rainbow scattering order, multiple-scattering sample and step counts, and a reserved value.
	glm::vec4 spectralBands[kSpectralBandCount]; ///< Per-band Rayleigh scale, solar scale, wavelength in nm, and ozone term.
	glm::vec4 cieXyz[kSpectralBandCount]; ///< Normalized CIE XYZ matching values for each spectral band.
	glm::vec4 sunDisk; ///< Sun solid angle, angular antialiasing bounds as cosines, and a reserved value.
	glm::vec4 mieBands[kSpectralBandCount]; ///< Per-band extinction/scattering scales for the two aerosol populations.
	glm::vec4 rainbowAxisX; ///< World-space local X axis for rainbow coordinates.
	glm::vec4 rainbowAxisZ; ///< World-space local Z axis for rainbow coordinates.
	glm::vec4 apparentSun; ///< Refracted sun direction in xyz and vertical refraction scale in w.
	glm::vec4 proposalScattering; ///< Wavelength-averaged scattering coefficients for path sampling.
	glm::vec4 proposalExtinction; ///< Wavelength-averaged extinction coefficients for path sampling.
};

static_assert(offsetof(SceneData, skyVrtParams) == 48);
static_assert(offsetof(SceneData, rainbowMultiple) == 112);
static_assert(offsetof(SceneData, spectralBands) == 128);
static_assert(offsetof(SceneData, proposalScattering) == 1008);
static_assert(offsetof(SceneData, proposalExtinction) == 1024);
static_assert(sizeof(SceneData) == 1040);

/// Descriptor types in the binding order expected by Main.comp.
constexpr std::array kDescriptorTypes { vk::DescriptorType::eStorageImage, vk::DescriptorType::eUniformBuffer, vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageBuffer };

/// Per-dispatch camera, frame, polarizer, and image parameters.
struct PushConstants {
	glm::vec4 forward; ///< Camera forward direction.
	glm::vec4 right; ///< Horizontal camera basis scaled by aspect ratio and field of view.
	glm::vec4 up; ///< Vertical camera basis scaled by field of view.
	glm::vec4 frame; ///< Frame index, exposure, accumulation-reset flag, and reserved value.
	glm::vec4 polarizer; ///< Enabled flag, linear angle, ellipticity, and reserved value.
	glm::uvec2 imageSize; ///< Swapchain width and height in pixels.
};

/// Owns the window, Vulkan objects, scene buffers, and render loop.
class App {
public:
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

	/// Packs configured atmospheric, spectral, and rainbow values for the shader.
	[[nodiscard]] SceneData BuildSceneData() const {
		const SkySpectralConfig& s = m_config.sky.spectral;
		const RainbowConfig& r = m_config.rainbow;
		SceneData sceneData {
			.skySpectralParams = { s.betaRayleigh550, s.aerosols[0].beta, 0.0f, s.sunRadiance550 },
			.skyRadiiScaleHeights = { s.earthRadius, s.atmosphereRadius, s.scaleHeightRayleigh, s.aerosols[0].scaleHeight },
			.skySunDirectionRadius = { s.sunDirection[0], s.sunDirection[1], s.sunDirection[2], s.sunRadius },
			.skyVrtParams = { s.sunAa, s.rayleighDepolarization, static_cast<float>(s.mieTableAngleBins), s.aerosols[1].scaleHeight },
			.rainbowRadiiEdge = { r.radii.x, r.radii.y, r.radii.z, r.edgeSoftness },
			.rainbowOptical = { r.scatteringCoefficient, r.extinctionCoefficient, static_cast<float>(r.angleBins), static_cast<float>(r.viewSteps) },
			.rainbowMultiple = { r.scatteringOrders, r.multipleScatteringSamples, r.multipleScatteringSteps, 0 },
		};

		float ySum = 0.0f;
		const double sun550 = ComputeSpectralAtWavelength(550.0).sunIrradianceScale;
		for (int band = 0; band < kSpectralBandCount; ++band) {
			const double wavelength = kSpectralLambdaMinNm + (kSpectralLambdaStepNm * band);
			const SpectralBand spectral = ComputeSpectralAtWavelength(wavelength);
			sceneData.spectralBands[band] = {
				static_cast<float>(s.betaRayleigh550 * spectral.betaRayleighScale),
				static_cast<float>(s.sunRadiance550 * spectral.sunIrradianceScale / sun550),
				static_cast<float>(kSpectralLambdaMinNm + (kSpectralLambdaStepNm * band)),
				static_cast<float>(spectral.ozoneCrossSection * s.ozoneDobsonUnits * 2.687e20 / 15000.0)
			};

			sceneData.cieXyz[band] = glm::vec4 { spectral.cie, 0.0 };
			ySum += sceneData.cieXyz[band][1];
		}

		for (auto& xyz : sceneData.cieXyz)
			for (int i = 0; i < 3; ++i) xyz[i] /= ySum;

		sceneData.sunDisk = { 2.0f * kPi * (1.0f - std::cos(s.sunRadius)), std::cos(s.sunRadius + s.sunAa), std::cos(s.sunRadius - s.sunAa), 0.0f };
		const float sunNorm = std::hypot(s.sunDirection[0], s.sunDirection[1], s.sunDirection[2]);
		const float trueAltitude = std::asin(s.sunDirection[1] / sunNorm) * 180.0f * std::numbers::inv_pi_v<float>;
		const auto refraction = [](float altitude) {
			const float h = std::max(altitude, -1.0f);
			return 1.02f / 60.0f / std::tan((h + (10.3f / (h + 5.11f))) * kPi / 180.0f);
		};

		const float apparentAltitude = (trueAltitude + refraction(trueAltitude)) * kPi / 180.0f;
		const float horizontal = std::hypot(s.sunDirection[0], s.sunDirection[2]);
		sceneData.apparentSun = { std::cos(apparentAltitude) * s.sunDirection[0] / horizontal, std::sin(apparentAltitude), std::cos(apparentAltitude) * s.sunDirection[2] / horizontal, 1.0f + ((refraction(trueAltitude + 0.1f) - refraction(trueAltitude - 0.1f)) / 0.2f) };
		const float sunLength = std::hypot(s.sunDirection[0], s.sunDirection[2]);
		const float ax = -s.sunDirection[0] / sunLength;
		const float az = -s.sunDirection[2] / sunLength;
		sceneData.rainbowCenterEnabled = { ax * r.distance, r.height, az * r.distance, static_cast<float>(r.enabled) };
		sceneData.rainbowAxisX = { az, 0.0f, -ax, 0.0f };
		sceneData.rainbowAxisZ = { ax, 0.0f, az, 0.0f };

		for (size_t aerosol = 0; aerosol < 2; ++aerosol) {
			const double beta = s.aerosols[aerosol].beta;
			const double reference = m_mieCrossSections[aerosol][6].x;
			for (int band = 0; band < kSpectralBandCount; ++band) {
				const glm::dvec2& cross = m_mieCrossSections[aerosol][static_cast<size_t>(band)];
				sceneData.mieBands[band][static_cast<int>(2 * aerosol)] = static_cast<float>(beta * cross.x / reference);
				sceneData.mieBands[band][static_cast<int>((2 * aerosol) + 1)] = static_cast<float>(beta * cross.y / reference);
			}
		}

		double proposalWeight = 0.0;
		glm::dvec3 scattering { 0.0 };
		glm::dvec4 extinction { 0.0 };
		for (int band = 0; band < kSpectralBandCount; ++band) {
			const auto& spectral = sceneData.spectralBands[band];
			const auto& mie = sceneData.mieBands[band];
			const double weight = spectral.y * glm::length(glm::dvec3(sceneData.cieXyz[band]));
			proposalWeight += weight;
			scattering += weight * glm::dvec3(spectral.x, mie.y, mie.w);
			extinction += weight * glm::dvec4(spectral.x, mie.x, spectral.w, mie.z);
		}

		if (proposalWeight > 0.0) {
			sceneData.proposalScattering = glm::vec4(scattering / proposalWeight, 0.0);
			sceneData.proposalExtinction = extinction / proposalWeight;
		}

		return sceneData;
	}

	/// Allocates a storage buffer and copies a lookup table into it.
	/// @tparam T Element type stored in the table.
	/// @param table Data to upload.
	/// @return The allocated and populated Vulkan buffer.
	template <typename T>
	[[nodiscard]] vma::raii::Buffer CreateTableBuffer(const std::vector<T>& table) const {
		vma::raii::Buffer buffer = CreateBuffer(table.size() * sizeof(T), vk::BufferUsageFlagBits::eStorageBuffer);
		UploadToBuffer(buffer, std::as_bytes(std::span { table }));
		return buffer;
	}

	/// Builds spectral lookup tables and uploads them with the scene uniform data.
	void CreateSceneBuffers() {
		const SkySpectralConfig& s = m_config.sky.spectral;
		m_sceneDataBuffer = CreateBuffer(sizeof(SceneData), vk::BufferUsageFlagBits::eUniformBuffer);
		std::vector<glm::vec4> mie = ComputeMieScatteringTable(s, 0, m_mieCrossSections[0]);
		std::ranges::copy(ComputeMieScatteringTable(s, 1, m_mieCrossSections[1]), std::back_inserter(mie));
		const int bins = std::max(2, static_cast<int>(s.mieTableAngleBins));
		std::array<double, kSpectralBandCount> visibleWeights { };
		for (int band = 0; band < kSpectralBandCount; ++band) {
			const double wavelength = kSpectralLambdaMinNm + (kSpectralLambdaStepNm * band);
			const SpectralBand spectral = ComputeSpectralAtWavelength(wavelength);
			visibleWeights[band] = spectral.sunIrradianceScale * glm::length(spectral.cie);
		}

		for (size_t aerosol = 0; aerosol < 2; ++aerosol) {
			auto weights = visibleWeights;
			for (int band = 0; band < kSpectralBandCount; ++band) weights[band] *= m_mieCrossSections[aerosol][band].y;
			AppendSamplingCdf(mie, bins, aerosol * static_cast<size_t>(kSpectralBandCount * bins), weights);
		}

		m_mieScatteringBuffer = CreateTableBuffer(mie);
		std::vector<glm::vec4> rainbow = ComputeRainbowScatteringTable(m_config.rainbow, s.sunRadius);
		AppendSamplingCdf(rainbow, static_cast<int>(m_config.rainbow.angleBins), 0, visibleWeights);
		m_rainbowScatteringBuffer = CreateTableBuffer(rainbow);
		m_transmittanceBuffer = CreateTableBuffer(ComputeTransmittanceTable(s));
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
		std::println("Controls: right-drag to look, R resets the view.");
		std::println("Polarizer: P toggles it, C switches linear/elliptical, [ ] adjusts it.");
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
								   .set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)
								   .add_fallback_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
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

		for (size_t i = 0; i < images.size(); ++i) {
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

	/// Creates the compute pipeline and supplies specialization and push-constant layouts.
	void CreatePipeline() {
		const vk::raii::ShaderModule computeModule { m_device, vk::ShaderModuleCreateInfo { { }, sizeof(kPathTracerSpirv), std::data(kPathTracerSpirv) } };
		const std::array<uint32_t, 5> settings = { std::max(1u, m_config.sky.spectral.scatteringOrders), m_config.sky.spectral.viewSteps, m_config.sky.spectral.samples, m_config.sky.spectral.secondarySamples, (m_config.rainbow.enabled != 0u) ? 1u : 0u };
		const std::array<vk::SpecializationMapEntry, 5> entries = { { { 0, 0, 4 }, { 1, 4, 4 }, { 2, 8, 4 }, { 3, 12, 4 }, { 4, 16, 4 } } };
		const vk::SpecializationInfo specialization { static_cast<uint32_t>(entries.size()), entries.data(), sizeof(settings), settings.data() };
		const vk::PushConstantRange pushRange { vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants) };
		vk::PipelineLayoutCreateInfo layoutInfo { };
		layoutInfo.setSetLayouts(*m_descriptorSetLayout);
		layoutInfo.setPushConstantRanges(pushRange);
		m_pipelineLayout = vk::raii::PipelineLayout(m_device, layoutInfo);
		const vk::PipelineShaderStageCreateInfo stage { { }, vk::ShaderStageFlagBits::eCompute, *computeModule, "main", &specialization };
		m_computePipeline = vk::raii::Pipeline(m_device, nullptr, vk::ComputePipelineCreateInfo { { }, stage, *m_pipelineLayout });
	}

	/// Creates per-frame command buffers and synchronization plus the accumulation buffer.
	void CreateFrameResources() {
		const uint32_t count = kFramesInFlight;
		vk::raii::CommandBuffers commandBuffers(m_device, { *m_commandPool, vk::CommandBufferLevel::ePrimary, count });
		const vk::FenceCreateInfo signaled { vk::FenceCreateFlagBits::eSignaled };
		m_frames.clear();

		for (uint32_t i = 0; i < count; ++i) m_frames.push_back({ .commandBuffer = std::move(commandBuffers[i]), .imageAvailable = m_device.createSemaphore({ }), .inFlight = m_device.createFence(signaled) });
		const vk::DeviceSize pixels = static_cast<vk::DeviceSize>(m_swapchainExtent.width) * m_swapchainExtent.height;
		m_accumulationBuffer = CreateBuffer(pixels * 4 * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer, { });
	}

	/// Builds shader parameters from the current view and resets accumulation after view changes.
	PushConstants BuildPushConstants() {
		const CameraController::View view = m_camera.GetView();
		const bool reset = m_lastView != view;
		m_lastView = view;
		if (reset) m_frameIndex = 0;

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
			.frame = { static_cast<float>(m_frameIndex), m_config.sky.exposure, reset ? 1.0f : 0.0f, 0.0 },
			.polarizer = { view.polarizerEnabled ? 1.0f : 0.0f, view.polarizerAngle, view.polarizerEllipticity, 0.0 },
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
		const vk::MemoryBarrier2 accumulation { Stage::eComputeShader, Access::eShaderStorageWrite, Stage::eComputeShader, Access::eShaderStorageRead | Access::eShaderStorageWrite };
		commandBuffer.pipelineBarrier2(vk::DependencyInfo { }.setMemoryBarriers(accumulation));
		imageBarrier(Stage::eNone, Access::eNone, Stage::eComputeShader, Access::eShaderStorageWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

		commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *m_computePipeline);
		const vk::DescriptorImageInfo imageInfo { { }, *m_swapchainImageViews[imageIndex], vk::ImageLayout::eGeneral };
		const auto bufferInfo = [](const vma::raii::Buffer& b) { return vk::DescriptorBufferInfo { *b, 0, vk::WholeSize }; };
		const std::array bufferInfos { bufferInfo(m_sceneDataBuffer), bufferInfo(m_mieScatteringBuffer), bufferInfo(m_rainbowScatteringBuffer), bufferInfo(m_transmittanceBuffer), bufferInfo(m_accumulationBuffer) };

		std::array<vk::WriteDescriptorSet, kDescriptorTypes.size()> writes { };
		writes[0] = vk::WriteDescriptorSet { { }, 0, 0, kDescriptorTypes[0], imageInfo };
		for (uint32_t i = 1; i < writes.size(); ++i) writes[i] = vk::WriteDescriptorSet { { }, i, 0, kDescriptorTypes[i], { }, bufferInfos[i - 1] };

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

		++m_frameIndex;
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

			++frames;
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

	GLFWwindow* m_window = nullptr;

	vk::raii::Context m_context;
	vkb::Instance m_selectedInstance { };
	vkb::PhysicalDevice m_selectedPhysicalDevice { };
	vkb::Device m_selectedDevice { };
	vkb::Swapchain m_selectedSwapchain { };
	vk::raii::Instance m_instance { nullptr };
	vk::raii::SurfaceKHR m_surface { nullptr };
	vk::raii::PhysicalDevice m_physicalDevice { nullptr };
	vk::raii::Device m_device { nullptr };
	uint32_t m_queueFamily = 0;
	vk::raii::Queue m_graphicsQueue { nullptr };
	vk::raii::Queue m_presentQueue { nullptr };

	vma::raii::Allocator m_allocator { nullptr };
	vma::raii::Buffer m_sceneDataBuffer { nullptr }, m_mieScatteringBuffer { nullptr }, m_rainbowScatteringBuffer { nullptr };
	vma::raii::Buffer m_transmittanceBuffer { nullptr }, m_accumulationBuffer { nullptr };
	std::array<std::array<glm::dvec2, kSpectralBandCount>, 2> m_mieCrossSections { };

	vk::raii::SwapchainKHR m_swapchain { nullptr };
	vk::Extent2D m_swapchainExtent { };
	std::vector<vk::Image> m_swapchainImages;
	std::vector<vk::raii::ImageView> m_swapchainImageViews;
	std::vector<vk::raii::Semaphore> m_renderFinished;

	vk::raii::DescriptorSetLayout m_descriptorSetLayout { nullptr };
	vk::raii::PipelineLayout m_pipelineLayout { nullptr };
	vk::raii::Pipeline m_computePipeline { nullptr };
	vk::raii::CommandPool m_commandPool { nullptr };

	std::vector<FrameResources> m_frames;
	uint32_t m_currentFrame = 0;
	uint64_t m_frameIndex = 0;
	float m_displayFps = 0.0f;
	RuntimeConfig m_config { };
	CameraController m_camera;
	std::optional<CameraController::View> m_lastView;
};

/// Application entry point. Reports setup and runtime exceptions to stderr.
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
