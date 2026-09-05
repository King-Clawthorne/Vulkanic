// Vulkanic — application entry point.
//
// Delegates the entire program to RunVulkanPathTracer(). Any std::exception
// escaping from initialization or the render loop is caught here, reported
// to stderr, and translated into a non-zero exit code so callers (build
// scripts, CI) can detect failure without inspecting log output.

#include "renderer/VulkanPathTracer.h"

#include <cstdio>
#include <exception>
#include <print>
#include <charconv>
#include <stdexcept>
#include <string_view>

int main(int argc, char** argv)
{
    try
    {
        BenchmarkOptions benchmark;
        for (int i = 1; i < argc; ++i)
        {
            const std::string_view option = argv[i];
            if (option == "--help")
            {
                std::println("Usage: Vulkanic [--benchmark FRAMES [--warmup FRAMES] [--config FILE] [--capture-hdr FILE]]");
                return 0;
            }
            if (option != "--benchmark" && option != "--warmup"
                && option != "--config" && option != "--capture-hdr")
                throw std::runtime_error("Unknown command-line option.");
            if (++i == argc) throw std::runtime_error("Missing command-line option value.");
            const std::string_view value = argv[i];
            if (option == "--config") benchmark.configPath = argv[i];
            else if (option == "--capture-hdr") benchmark.captureHdrPath = argv[i];
            else
            {
                uint32_t count = 0;
                const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), count);
                if (error != std::errc{} || end != value.data() + value.size()
                    || (option == "--benchmark" && count == 0))
                    throw std::runtime_error("Frame count must be an unsigned integer; benchmark frames must be positive.");
                if (option == "--benchmark") benchmark.frames = count;
                else benchmark.warmupFrames = count;
            }
        }
        if (argc > 1 && benchmark.frames == 0)
            throw std::runtime_error("Benchmark options require --benchmark FRAMES.");
        RunVulkanPathTracer(benchmark);
        return 0;
    }
    catch (const std::exception& error)
    {
        // Top-level fatal error sink — anything thrown by the tracer
        // surfaces here so the user sees a single, consistent diagnostic.
        std::println(stderr, "[ERROR] {}", error.what());
        return 1;
    }
}
