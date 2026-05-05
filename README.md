# Vulkanic

A lightweight, purely native C++17 Vulkan path tracer demonstrating hardware-accelerated ray tracing on Windows. **Vulkanic** avoids heavy third-party framework dependencies, showing how to build a path tracer using the Vulkan API and the Win32 API from scratch.

---

## Project Overview (STAR)

### Situation

Most open-source Vulkan ray-tracing samples are either trapped inside large, opinionated frameworks (engine repos, vendor SDK demo suites) or layered on top of GLFW + GLM + TinyObjLoader + a JSON library. That stack makes it hard to see *which* lines actually drive the GPU and which are framework glue. Newcomers learning `VK_KHR_ray_tracing_pipeline` end up reading wrappers around wrappers before they ever see a `vkCmdTraceRaysKHR` call.

### Task

Build a complete, runnable Monte Carlo path tracer that:

- Targets the raw Vulkan 1.2 ray-tracing extensions on Windows.
- Has **zero** third-party wrappers — no GLFW, no GLM, no TinyObjLoader, no nlohmann::json.
- Stays small enough to read end-to-end in an afternoon.
- Still produces a physically plausible image (GGX BRDF, conductor Fresnel, energy compensation, atmospheric sky, ACES tonemap).

### Action

The repository implements the full pipeline from the OS layer up:

- **Hardware-Accelerated Ray Tracing** — `VK_KHR_ray_tracing_pipeline` with ray-generation (`path_tracer.rgen`), closest-hit (`path_tracer.rchit`), primary miss (`path_tracer.rmiss`), and shadow miss (`shadow.rmiss`) shaders. BLAS per mesh, one TLAS for the scene.
- **Advanced Material Rendering** — anisotropic GGX with Heitz 2018 VNDF importance sampling, complex-IOR (conductor) Fresnel, and the Fdez-Aguera multiscatter energy-compensation term so rough surfaces stay bright.
- **Direct Sun Lighting** — next-event estimation against a configurable sun cone, with TerminateOnFirstHit + SkipClosestHit shadow rays for cheap occlusion.
- **Compute Shader Atmosphere** — Rayleigh + Mie single-scattering sky integrated in `sky.comp` and shared with the miss shader for image-based lighting.
- **Native Asset Loaders** — a hand-written `.obj` parser (`ObjModel.cpp`) and a hand-written JSON parser (`RuntimeConfig.cpp`) load the scene and the tunable parameters at startup.
- **Hot Config Reload** — `path_tracer_config.json` is polled each frame; edits to materials, instances, sky, or camera apply without restarting.
- **Shader Debugging** — `VK_KHR_shader_non_semantic_info` and unoptimized GLSL→SPIR-V output keep RenderDoc / Nsight legible.

### Result

A single ~5,000-line repository containing every layer needed to render a physically based scene with hardware ray tracing:

- One executable, one config file, one models folder. Run `build.ps1` and you have a moving camera over a path-traced scene.
- The Vulkan code is organized so that each major step (instance → device → swapchain → BLAS/TLAS → pipeline → SBT → render loop) is a single, self-contained method on `VulkanApp` — searchable via the `SECTION:` banners in `VulkanPathTracer.cpp`.
- Every shader file documents the algorithm it implements at the top, so the closest-hit shader reads as "what BSDF/sampling/RR we do" rather than just GLSL noise.
- The whole project is a usable reference for anyone porting another renderer to Vulkan ray tracing or learning the extension surface without a framework in the way.

---

## System Requirements

- **OS:** Windows 10 / 11 (uses `VK_USE_PLATFORM_WIN32_KHR`).
- **GPU:** Vulkan 1.2+ with hardware ray tracing (e.g. NVIDIA RTX or AMD RX 6000 series).
- **Vulkan SDK:** installed and exposed via the `VULKAN_SDK` environment variable.
- **Compiler:** any C++17 toolchain — MSVC via Visual Studio is the tested path.
- **CMake:** 3.20 or newer.

## Building

The provided PowerShell script sets up the MSVC environment and (optionally) `sccache`:

```powershell
# from the project root, in a PowerShell terminal
.\build.ps1
```

For a clean rebuild:

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1 -Clean
```

Or build manually with CMake:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Project Structure

- **Shaders (`glslc` → SPIR-V):**
  - `path_tracer.rgen` — ray generation (camera rays + tonemap + write).
  - `path_tracer.rmiss` — primary-ray miss (sky lookup).
  - `shadow.rmiss` — shadow-ray miss (marks the path as unoccluded).
  - `path_tracer.rchit` — closest-hit shader (BSDF, NEE, Russian roulette, recursion).
  - `sky.comp` — atmospheric sky model included by every stage.
  - `path_tracer_common.glsl` — descriptor bindings, payloads, RNG, math.
- **C++ Core:**
  - `VulkanPathTracer.h` / `.cpp` — Vulkan setup, swapchain, ray-tracing pipeline, Win32 window + message loop.
  - `ObjModel.h` / `.cpp` — self-contained `.obj` mesh parser.
  - `RuntimeConfig.h` / `.cpp` — JSON parser + scene/runtime configuration.
  - `main.cpp` — entry point.
- **Assets:**
  - `models/box.obj` / `models/box.mtl` — default mesh used by the bundled config.
- **Build & Configuration:**
  - `build.ps1` — PowerShell wrapper that loads the MSVC environment and runs CMake.
  - `CMakeLists.txt` — CMake project; also drives GLSL → SPIR-V compilation.
  - `path_tracer_config.json` — materials, models, instances, sky parameters, camera, render settings.
