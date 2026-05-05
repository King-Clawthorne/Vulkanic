# Vulkanic

A lightweight, purely native C++17 Vulkan path tracer demonstrating hardware-accelerated ray tracing on Windows. **Vulkanic** avoids heavy third-party framework dependencies, showcasing how to build a path tracer using the Vulkan API and the Win32 API from scratch.

## Features

- **Hardware-Accelerated Ray Tracing:** Implements `VK_KHR_ray_tracing_pipeline` for ray generation (`.rgen`), closest hit (`.rchit`), and miss (`.rmiss`) shaders.
- **Advanced Material Rendering:** Includes support for multiscatter GGX energy compensation (Fdez-Aguera approximation) and Heitz 2018 anisotropic VNDF sampling.
- **Zero Third-Party Wrapper Clutter:** Built directly on top of the Vulkan SDK and the native Win32 API. Does not rely on libraries like GLFW, GLM, or TinyObjLoader—it uses its own mathematical structures and OBJ parser.
- **Compute Shader Environment:** Renders the sky and background environments dynamically utilizing Vulkan compute shaders (`sky.comp`).
- **Dynamic Configuration:** Supports runtime configurations mapped from a lightweight `path_tracer_config.json` model.
- **Native `.obj` Loading:** Parses and loads standard 3D `.obj` models internally.
- **Shader Debugging Support:** Utilizes `VK_KHR_shader_non_semantic_info` extension and emits unoptimized shaders with debug symbols for easier troubleshooting.

## System Requirements

- **OS:** Windows 10 / 11 (Requires `VK_USE_PLATFORM_WIN32_KHR`).
- **GPU:** A Vulkan 1.2+ capable GPU with hardware ray tracing support (e.g., NVIDIA RTX or AMD RX 6000 series).
- **Vulkan SDK:** Must be installed and registered in your environment parameters (`VULKAN_SDK`).
- **Compiler:** C++17 compatible compiler (e.g., MSVC via Visual Studio).
- **CMake:** Version 3.20 or newer.

## Building

You can quickly build and configure the project using the provided PowerShell build script, which maps MSVC variables and optionally uses `sccache`:

```powershell
# Open a PowerShell terminal in the project directory
.\build.ps1
```

To run a clean build, simply append the `-Clean` parameter:
```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1 -Clean
```

**Manual CMake Build:**

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Project Structure

- **Shaders (`glslc` compiled to SPIR-V):**
  - `path_tracer.rgen` - Ray generation shader.
  - `path_tracer.rmiss` - Primary ray miss shader.
  - `shadow.rmiss` - Shadow ray miss shader; marks a ray as unoccluded when no geometry is hit.
  - `path_tracer.rchit` - Ray closest hit shader.
  - `sky.comp` - Environment compute shader.
  - `path_tracer_common.glsl` - Shared shader utilities, mathematical constants, and payload structures.
- **C++ Core:**
  - `VulkanPathTracer.h` / `.cpp` - Handles Vulkan setup, initialization, synchronization, ray-tracing pipelines, and the Win32 message loop.
  - `ObjModel.h` / `.cpp` - Dedicated internal `.obj` 3D mesh parser.
  - `RuntimeConfig.h` / `.cpp` - Parsing mapping for handling application specifications and attributes from `path_tracer_config.json`.
  - `main.cpp` - Execution entrypoint.
- **Assets:**
  - `models/box.obj` / `models/box.mtl` - Bundled default box mesh used by the example scene configuration.
- **Build & Configuration:**
  - `build.ps1` - PowerShell script mapping MSVC environments to build with `cmake`.
  - `CMakeLists.txt` - CMake configuration linking native modules and compiling GLSL to SPIR-V.
  - `path_tracer_config.json` - Defines materials, objects, sky spectral constants, camera properties, and render settings.
