#include <VkBootstrap.h>
#include <vulkan/vulkan_raii.hpp>
#define VMA_IMPLEMENTATION
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numbers>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vk_mem_alloc_raii.hpp>

namespace {
	constexpr float kEarthRadius = 6'360'000.0f;
	constexpr float kAtmosphereRadius = 6'420'000.0f;
	constexpr float kScaleHeight = 7'994.0f;
	constexpr float kDepolarization = 0.0279f;
	constexpr float kGreenBeta = 13.5e-6f;
	constexpr float kSunRadius = 0.00465f;
	constexpr float kSunRadiance = 8'317.742f;
	constexpr uint32_t kViewPanels = 48;
	constexpr uint32_t kWarmupRuns = 3;
	constexpr uint32_t kMeasuredRuns = 21;
	constexpr uint32_t kReplicasPerScenario = 16;

	// These host structures mirror the uniform and storage-buffer layouts in Research.comp.
	struct alignas(16) SceneData {
		std::array<float, 4> skyParams;
		std::array<float, 4> skySunDirectionRadius;
		std::array<float, 4> rayleigh;
		std::array<float, 4> sunParams;
	};

	struct alignas(16) ResearchRay {
		std::array<float, 4> origin;
		std::array<float, 4> viewDir;
		std::array<float, 4> sunDir;
	};

	struct alignas(16) ResearchResult {
		std::array<std::array<float, 4>, 3> rgb;
		std::array<float, 4> diagnostics;
	};

	struct PushConstants {
		uint32_t rayCount;
		uint32_t viewSamples;
		uint32_t maxAzimuth;
		uint32_t method;
		float tolerance;
	};

	struct Scenario {
		float sunElevation;
		float viewElevation;
		bool heldout;
	};

	struct Configuration {
		std::string label;
		uint32_t method;
		uint32_t azimuth;
		float tolerance;
	};

	template <typename T>
	T unwrap(vkb::Result<T> result, const char* operation) {
		if (!result) throw std::runtime_error(std::string(operation) + ": " + result.error().message());
		return std::move(result.value());
	}

// CMake adds the build directory to the include path and makes this generated
// SPIR-V file a prerequisite of the compilation that expands #embed.
alignas(uint32_t) const unsigned char kShaderBytes[] = {
#embed "Research.comp.spv"
};

	float radians(float degrees) { return degrees * (std::numbers::pi_v<float> / 180.0f); }

	ResearchRay make_ray(const Scenario& scenario) {
		const float sun = radians(scenario.sunElevation);
		const float view = radians(scenario.viewElevation);
		const float viewAzimuth = radians(90.0f);

		ResearchRay ray { };
		ray.origin = { 0.0f, kEarthRadius + 1.0f, 0.0f, 0.0f };
		ray.viewDir = { std::cos(view) * std::cos(viewAzimuth), std::sin(view), std::cos(view) * std::sin(viewAzimuth), 0.0f };
		ray.sunDir = { std::cos(sun), std::sin(sun), 0.0f, 0.0f };

		return ray;
	}
} // namespace

int main(int argc, char** argv) {
	const std::filesystem::path outputPath = argc > 1 ? argv[1] : "results/gpu_sweep.csv";
	if (!outputPath.parent_path().empty()) std::filesystem::create_directories(outputPath.parent_path());

	vk::raii::Context context;

	// Use a headless Vulkan 1.4 device with push-descriptor support.
	auto bootstrapInstance = unwrap(vkb::InstanceBuilder().set_app_name("Vulkanic polarized integration research").set_engine_name("Vulkanic").require_api_version(1, 4, 0).set_headless().build(), "create instance");

	vk::raii::Instance instance { context, bootstrapInstance.instance };
	auto physical = unwrap(vkb::PhysicalDeviceSelector(bootstrapInstance).set_minimum_version(1, 4).select(), "select physical device");
	vk::raii::PhysicalDevice physicalDevice { instance, physical.physical_device };
	std::cout << "GPU: " << physical.properties.deviceName << '\n';
	auto bootstrapDevice = unwrap(vkb::DeviceBuilder(physical).build(), "create device");
	auto [computeQueue, queueFamily] = unwrap(bootstrapDevice.get_queue_and_index(vkb::QueueType::compute), "get compute queue");
	vk::raii::Device device { physicalDevice, bootstrapDevice.device };
	vk::raii::Queue queue { device, computeQueue };

	vma::raii::Allocator allocator { instance, device, vma::AllocatorCreateInfo { }.setPhysicalDevice(*physicalDevice).setVulkanApiVersion(VK_API_VERSION_1_4) };

	// Pilot cases come first. Held-out cases are marked in the final column.
	const std::array<Scenario, 20> scenarios { { { -8.0f, 0.0f, false }, { -5.0f, 0.0f, false }, { -5.0f, 10.0f, false }, { -2.0f, 0.0f, false }, { 0.0f, 0.0f, false }, { 0.0f, 45.0f, false }, { 10.0f, 0.0f, false }, { 20.0f, 90.0f, false }, { -9.0f, 0.0f, true }, { -8.0f, 10.0f, true }, { -7.0f, 0.0f, true }, { -6.0f, 10.0f, true }, { -4.0f, 0.0f, true }, { -3.0f, 10.0f, true }, { -2.0f, 45.0f, true }, { -1.0f, 0.0f, true }, { 1.0f, 0.0f, true }, { 2.0f, 10.0f, true }, { 5.0f, 30.0f, true }, { 15.0f, 45.0f, true } } };

	// Replicate the scenario set to keep the GPU workload large and uniform.
	auto rays = std::views::repeat(scenarios, kReplicasPerScenario) | std::views::join | std::views::transform([](const Scenario& s) { return make_ray(s); }) | std::ranges::to<std::vector>();

	// Mapped, coherent allocations allow direct host writes and result reads.
	const auto makeBuffer = [&allocator](vk::DeviceSize size, vk::BufferUsageFlags usage) { return allocator.createBuffer(vk::BufferCreateInfo { }.setSize(size).setUsage(usage), vma::AllocationCreateInfo { }.setUsage(vma::MemoryUsage::eAuto).setFlags(vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessRandom).setRequiredFlags(vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)); };

	auto inputBuffer = makeBuffer(sizeof(ResearchRay) * rays.size(), vk::BufferUsageFlagBits::eStorageBuffer);
	auto sceneBuffer = makeBuffer(sizeof(SceneData), vk::BufferUsageFlagBits::eUniformBuffer);
	auto outputBuffer = makeBuffer(sizeof(ResearchResult) * rays.size(), vk::BufferUsageFlagBits::eStorageBuffer);
	std::copy(rays.begin(), rays.end(), static_cast<ResearchRay*>(inputBuffer.getAllocation().getInfo().pMappedData));

	auto* scene = static_cast<SceneData*>(sceneBuffer.getAllocation().getInfo().pMappedData);
	scene->skyParams = { kEarthRadius, kAtmosphereRadius, kScaleHeight, kDepolarization };
	scene->skySunDirectionRadius = { 1.0f, 0.0f, 0.0f, kSunRadius };
	scene->rayleigh = { kGreenBeta * 0.512f, kGreenBeta, kGreenBeta * 2.231f, 0.0f };
	scene->sunParams = { 2.0f * std::numbers::pi_v<float> * (1.0f - std::cos(kSunRadius)), kSunRadiance, 0.0f, 0.0f };

	// Bind input rays, shared scene data, and output results in shader binding order.
	const std::array<vk::DescriptorSetLayoutBinding, 3> bindings { { { 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr }, { 1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr }, { 2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr } } };
	vk::raii::DescriptorSetLayout descriptorLayout { device, vk::DescriptorSetLayoutCreateInfo { }.setFlags(vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR).setBindings(bindings) };
	const std::array<vk::DescriptorBufferInfo, 3> bufferInfos { { { *inputBuffer, 0, VK_WHOLE_SIZE }, { *sceneBuffer, 0, VK_WHOLE_SIZE }, { *outputBuffer, 0, VK_WHOLE_SIZE } } };

	const std::array<vk::WriteDescriptorSet, 3> writes { { { { }, bindings[0].binding, 0, bindings[0].descriptorCount, bindings[0].descriptorType, nullptr, &bufferInfos[0] }, { { }, bindings[1].binding, 0, bindings[1].descriptorCount, bindings[1].descriptorType, nullptr, &bufferInfos[1] }, { { }, bindings[2].binding, 0, bindings[2].descriptorCount, bindings[2].descriptorType, nullptr, &bufferInfos[2] } } };
	vk::raii::PipelineLayout pipelineLayout { device, vk::PipelineLayoutCreateInfo { }.setSetLayouts(*descriptorLayout).setPushConstantRanges(vk::PushConstantRange { vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants) }) };

	const auto* shaderCode = reinterpret_cast<const uint32_t*>(kShaderBytes);
	vk::raii::ShaderModule shaderModule { device, vk::ShaderModuleCreateInfo { }.setCodeSize(sizeof(kShaderBytes)).setPCode(shaderCode) };
	vk::raii::Pipeline pipeline { device, nullptr, vk::ComputePipelineCreateInfo { }.setStage(vk::PipelineShaderStageCreateInfo { }.setStage(vk::ShaderStageFlagBits::eCompute).setModule(*shaderModule).setPName("main")).setLayout(*pipelineLayout) };

	vk::raii::CommandPool commandPool { device, vk::CommandPoolCreateInfo { }.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer).setQueueFamilyIndex(queueFamily) };
	auto commandBuffers = device.allocateCommandBuffers({ *commandPool, vk::CommandBufferLevel::ePrimary, 1 });
	auto& commandBuffer = commandBuffers.front();

	vk::raii::Fence fence { device, vk::FenceCreateInfo { } };
	const std::array<Configuration, 10> configurations { { { "point", 0, 8, 1.0e-3f }, { "fixed", 1, 8, 0.0f }, { "fixed", 1, 16, 0.0f }, { "fixed", 1, 32, 0.0f }, { "fixed", 1, 64, 0.0f }, { "fixed", 1, 128, 0.0f }, { "fixed", 1, 256, 0.0f }, { "adaptive", 2, 256, 1.0e-2f }, { "adaptive", 2, 256, 1.0e-3f }, { "adaptive", 2, 256, 1.0e-4f } } };

	std::ofstream csv(outputPath, std::ios::trunc);
	if (!csv) throw std::runtime_error("Cannot create output CSV: " + outputPath.string());
	csv << "sun_elevation_deg,view_elevation_deg,split,method,azimuth_count,tolerance,work_nodes,max_selected_azimuth,max_local_error_estimate,red_I,red_Q,red_U,red_V,green_I,green_Q,green_U,green_V,blue_I,blue_Q,blue_U,blue_V,ray_count,view_panels,view_evaluations,profile_repetitions,gpu_name\n";

	for (const Configuration& configuration : configurations) {
		const PushConstants push { static_cast<uint32_t>(rays.size()), kViewPanels, configuration.azimuth, configuration.method, configuration.tolerance };

		// Re-record the single command buffer for every warmup and measured run.
		for (auto _ : std::views::indices(kWarmupRuns + kMeasuredRuns)) {
			commandBuffer.reset();
			commandBuffer.begin(vk::CommandBufferBeginInfo { });
			commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
			commandBuffer.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, writes);
			commandBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
			commandBuffer.dispatch(static_cast<uint32_t>((rays.size() + 63) / 64), 1, 1);

			// Make shader writes visible to the host before reading mapped results.
			commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, { }, vk::MemoryBarrier { }.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eHostRead), { }, { });
			commandBuffer.end();

			device.resetFences(*fence);
			queue.submit(vk::SubmitInfo { }.setCommandBuffers(*commandBuffer), *fence);
			if (device.waitForFences(*fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) throw std::runtime_error("vkWaitForFences failed");
		}

		const auto* results = static_cast<const ResearchResult*>(outputBuffer.getAllocation().getInfo().pMappedData);

		for (auto [scenario, value] : std::views::zip(scenarios, std::span(results, scenarios.size()))) {
			csv << scenario.sunElevation << ',' << scenario.viewElevation << ',' << (scenario.heldout ? "heldout" : "pilot") << ',' << configuration.label << ',' << configuration.azimuth << ',' << configuration.tolerance << ',';

			// The fourth diagnostics lane is padding and has no CSV column.
			for (size_t component = 0; component < value.diagnostics.size() - 1; ++component) csv << value.diagnostics[component] << ',';
			std::ranges::copy(value.rgb | std::views::join, std::ostream_iterator<float>(csv, ","));

			csv << rays.size() << ',' << kViewPanels << ',' << (4 * kViewPanels) << ',' << (kWarmupRuns + kMeasuredRuns) << ",\"" << physical.properties.deviceName << "\"\n";
		}

		std::cout << configuration.label << " az=" << configuration.azimuth << " tol=" << configuration.tolerance << '\n';
	}

	std::cout << "Wrote " << outputPath.string() << '\n';
	return 0;
}
