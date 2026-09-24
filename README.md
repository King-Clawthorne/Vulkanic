# Vulkanic

Vulkanic is a real-time Vulkan path tracer for rendering a spectrally modeled sky. It computes atmospheric Rayleigh and Mie scattering, ozone absorption, a solar disk, and an optional rainbows, then displays the result in a GLFW window.

## Requirements

- CMake 3.30 or newer
- Ninja
- Clang 20 or newer with `clang-cl` (the included preset targets Windows)
- Vulkan SDK with Vulkan 1.4 support and `glslc`
- A Vulkan 1.4 capable GPU and driver

CMake downloads GLFW, GLM, Vulkan Memory Allocator-Hpp, and vk-bootstrap through FetchContent.

## Build

From PowerShell, run:

```powershell
./build.ps1
```

The script configures and builds the Release target with the `default` CMake preset. The executable is written to `build/Vulkanic.exe`.

To build manually:

```powershell
cmake --preset default
cmake --build --preset default --parallel
```

## Run

```powershell
./build/Vulkanic.exe
```

The window title reports the current frame rate and frame time. Controls:

- **W/A/S/D**: move forward, left, backward, and right
- **Space / Left Ctrl**: move up and down
- **Right mouse button + mouse**: look around
- **[ / ]**: rotate the polarizer
- **P**: toggle the polarizer
- **C**: toggle elliptical polarization display
- **R**: reset the camera and polarization controls
- **Escape**: close the window

## Project layout

- `src/` — Vulkan renderer and CPU-side sky/scattering tables
- `shaders/pathTracer.comp` — Vulkan compute shader
- `build.ps1` — clean configure and build helper
- `CMakeLists.txt` — dependencies, shader compilation, and executable target

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
