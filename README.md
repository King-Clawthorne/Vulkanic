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
- **Polarized Atmosphere** — Rayleigh + Lorenz-Mie sky rendering in `sky.comp`, including a CPU-baked Mie scattering-matrix table and Stokes-vector transport for camera-visible sky rays.
- **Runtime Polarization Filter** — press `P` to enable the camera analyzer, `C` to switch linear/elliptical mode, and use `[` / `]` to rotate the linear axis or adjust elliptical handedness/strength.
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

## Runtime Controls

- Hold right mouse button and move the mouse to look around.
- `W` / `A` / `S` / `D` move horizontally.
- `Q` / `E` or `Ctrl` / `Space` move down/up.
- Hold `Shift` for fast movement.
- Arrow keys look around without the mouse.
- `R` resets the camera to the JSON `initialPosition` / `initialLookAt`.
- `P` toggles the polarization filter.
- `C` switches the filter between linear and elliptical modes.
- In linear mode, `[` / `]` rotate the analyzer axis.
- In elliptical mode, `[` / `]` adjust ellipticity from left-circular through linear to right-circular.

Input rates live in `path_tracer_config.json` under `input`, including `moveSpeed`, `fastMoveSpeed`, `mouseSensitivity`, `keyLookSpeed`, and `polarizerRotateSpeed`.

## Runtime Configuration

`path_tracer_config.json` drives the scene and most renderer parameters:

- `render` sets resolution, swapchain frame count, samples per pixel, and maximum path depth.
- `camera` sets the startup view and vertical field of view.
- `input` controls movement, look speed, mouse sensitivity, and polarization adjustment speed.
- `sky.spectralConstants` controls the atmospheric model, sun, Rayleigh depolarization, aerosol parameters, Mie table resolution, and scattering orders.
- `models` maps scene model names to `.obj` files. Files are resolved from the repo's `models/` folder.
- `materials` defines per-material albedo, emission, conductor IOR (`eta`), and extinction (`k`).
- `instances` places models in the TLAS with independent material, position, rotation, and scale.

Most edits hot-reload while the app is running. Width, height, and `frameCount` are read at startup and require a restart.

## Project Structure

- **Shaders (`glslc` → SPIR-V):**
  - `path_tracer.rgen` — ray generation (camera rays + tonemap + write).
  - `path_tracer.rmiss` — primary-ray miss (polarized sky on camera rays, cheaper sky on indirect rays).
  - `shadow.rmiss` — shadow-ray miss (marks the path as unoccluded).
  - `path_tracer.rchit` — closest-hit shader (BSDF, NEE, Russian roulette, recursion).
  - `sky.comp` — Rayleigh/Mie atmosphere and Stokes-vector sky transport included by every stage.
  - `path_tracer_common.glsl` — descriptor bindings, payloads, RNG, math.
- **C++ Core:**
  - `VulkanPathTracer.h` / `.cpp` — Vulkan setup, swapchain, ray-tracing pipeline, Win32 window + message loop.
  - `ObjModel.h` / `.cpp` — self-contained `.obj` mesh parser.
  - `MieScattering.h` / `.cpp` — CPU Lorenz-Mie scattering-matrix precompute for the polarized sky.
  - `RuntimeConfig.h` / `.cpp` — JSON parser + scene/runtime configuration.
  - `main.cpp` — entry point.
- **Assets:**
  - `models/box.obj` / `models/box.mtl` — default mesh used by the bundled config.
- **Build & Configuration:**
  - `build.ps1` — PowerShell wrapper that loads the MSVC environment and runs CMake.
  - `CMakeLists.txt` — CMake project; also drives GLSL → SPIR-V compilation.
  - `path_tracer_config.json` — materials, models, instances, sky parameters, camera, render settings.
