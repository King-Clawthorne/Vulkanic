// Vulkanic — application entry point.
//
// Delegates the entire program to RunVulkanPathTracer(). Any std::exception
// escaping from initialization or the render loop is caught here, reported
// to stderr, and translated into a non-zero exit code so callers (build
// scripts, CI) can detect failure without inspecting log output.

#include "VulkanPathTracer.h"

#include <cstdio>
#include <exception>

int main()
{
    try
    {
        RunVulkanPathTracer();
        return 0;
    }
    catch (const std::exception& error)
    {
        // Top-level fatal error sink — anything thrown by the tracer
        // surfaces here so the user sees a single, consistent diagnostic.
        std::fprintf(stderr, "[ERROR] %s\n", error.what());
        return 1;
    }
}
