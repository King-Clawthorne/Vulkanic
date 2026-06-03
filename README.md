# Vulkanic

A lightweight, purely native C++17 Vulkan **real-time polarized-sky simulator**. **Vulkanic**
renders the daytime sky as a full Stokes-vector vector-radiative-transfer problem — Rayleigh + Mie
scattering, an ozone layer, ground coupling, and converged multiple scattering — then views it
through a runtime camera analyzer that switches between linear and elliptical polarization.

---

## Project Overview (STAR)

### Situation

Most renderers treat polarization as a post-process trick or ignore it entirely, and the sky as a
flat gradient or scalar single-scattering model. Scientific polarized-sky models usually live
offline in radiative-transfer packages. Vulkanic bridges that gap: a readable, dependency-free
Vulkan program that interactively explores *physically motivated* sky polarization and how an ideal
analyzer responds to it.

### Task

Build a complete, runnable real-time simulator that:

- Targets the raw Vulkan 1.2 ray-tracing pipeline on Windows.
- Has **zero** third-party wrappers — no GLFW, no GLM, no TinyObjLoader, no nlohmann::json.
- Carries the full Stokes vector (I, Q, U, V) through the atmosphere, not just scalar RGB radiance.
- Lets the user aim the camera around the sky dome and switch between linear and elliptical analyzer
  modes at runtime.

### Action

The repository implements the whole stack from the OS layer up:

- **Polarized vector radiative transfer** (`sky.comp`) — Rayleigh scattering with molecular
  depolarization, a CPU-baked **Lorenz–Mie** aerosol scattering matrix, a Chappuis-band **ozone**
  absorption layer, **ground ↔ sky coupling** (Lambertian reflection of direct sun *and* downwelling
  skylight), and **converged multiple scattering** via a backward polarized Monte-Carlo random walk.
  Everything is transported as Stokes vectors with proper Mueller matrices and frame rotations.
- **Runtime polarization analyzer** — an ideal elliptical analyzer applied per band in the
  ray-generation shader; `P` enables it, `C` switches linear/elliptical, `[` / `]` rotate the axis
  or sweep ellipticity.
- **From-scratch numerics** — a hand-written Lorenz–Mie solver (`MieScattering.cpp`) bakes the
  scattering-matrix table at startup; a hand-written JSON parser (`RuntimeConfig.cpp`) loads the
  tunable parameters.
- **Hot config reload** — `path_tracer_config.json` is polled each frame; sky/aerosol edits apply
  live (aerosol changes rebuild the Mie table).
- **Hardware ray-tracing pipeline** — the renderer keeps the `VK_KHR_ray_tracing_pipeline` and runs
  a single ray-generation shader that evaluates the sky analytically per pixel (there is no scene
  geometry, so no BLAS/TLAS, hit, or miss shaders).

### Result

A compact repository whose only job is to render the polarized sky and let you study it:

- One executable, one config file. Run `build.ps1` and you can look around a physically motivated
  polarized sky.
- The analyzer toggles live between linear and elliptical, making the renderer useful for studying
  the Rayleigh polarization band, neutral points, and analyzer response.
- Each shader and module documents the model it implements at the top.

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

- Hold right mouse button and move the mouse to look around the sky.
- Arrow keys look around without the mouse. `R` resets the camera to the JSON view.
- `P` toggles the polarization analyzer.
- `C` switches the analyzer between linear and elliptical modes.
- In linear mode, `[` / `]` rotate the analyzer axis.
- In elliptical mode, `[` / `]` sweep ellipticity from left-circular through linear to right-circular.

(WASD/Q/E movement still works but the sky is directional, so position has no visible effect.)

## Runtime Configuration

`path_tracer_config.json` drives the renderer:

- `render` — resolution, swapchain frame count, samples per pixel.
- `camera` — startup view and vertical field of view.
- `input` — look speed, mouse sensitivity, analyzer rotation speed.
- `sky.spectralConstants` — the atmospheric model: Rayleigh/Mie coefficients, sun, Rayleigh
  depolarization, aerosol (Lorenz–Mie) parameters, Mie table resolution, scattering orders, ozone
  absorption + layer profile, and ground albedo.

Most edits hot-reload while running. Width, height, and `frameCount` are read at startup.

## Project Structure

- **Shaders (`glslc` → SPIR-V):**
  - `path_tracer.rgen` — the whole renderer: camera ray per pixel → polarized sky → analyzer →
    tonemap → image.
  - `sky.comp` — the polarized VRT atmosphere (Rayleigh/Mie/ozone, ground coupling, converged
    multiple scattering, Stokes machinery).
  - `path_tracer_common.glsl` — descriptor bindings, push constants, RNG, tonemap.
- **C++ Core:**
  - `VulkanPathTracer.h` / `.cpp` — Vulkan setup, swapchain, ray-tracing pipeline, Win32 window +
    message loop.
  - `MieScattering.h` / `.cpp` — CPU Lorenz–Mie scattering-matrix precompute for the polarized sky.
  - `RuntimeConfig.h` / `.cpp` — JSON parser + runtime configuration.
  - `main.cpp` — entry point.
- **Build & Configuration:**
  - `build.ps1` — PowerShell wrapper that loads the MSVC environment and runs CMake.
  - `CMakeLists.txt` — CMake project; also drives GLSL → SPIR-V compilation.
  - `path_tracer_config.json` — sky parameters, camera, input, render settings.
