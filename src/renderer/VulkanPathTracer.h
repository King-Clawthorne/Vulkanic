#pragma once

#include <cstdint>
#include <filesystem>

struct BenchmarkOptions
{
    uint32_t frames = 0; // Zero keeps the normal interactive renderer.
    uint32_t warmupFrames = 60;
    std::filesystem::path configPath;
    std::filesystem::path captureHdrPath;
};

// Public entry point for the Vulkanic path tracer. Implemented in
// VulkanPathTracer.cpp; owns the Win32 window, Vulkan device, compute
// pipeline, and the message-loop / render-loop. Throws std::exception on
// fatal initialization or runtime failure so main() can report it.
void RunVulkanPathTracer(const BenchmarkOptions& benchmark = {});
