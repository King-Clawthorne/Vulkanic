#pragma once

// Public entry point for the Vulkanic path tracer. Implemented in
// VulkanPathTracer.cpp; owns the Win32 window, Vulkan device, ray-tracing
// pipeline, and the message-loop / render-loop. Throws std::exception on
// fatal initialization or runtime failure so main() can report it.
void RunVulkanPathTracer();
