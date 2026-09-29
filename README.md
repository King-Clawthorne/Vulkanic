# Vulkanic

Vulkanic is a real-time Vulkan path tracer for exploring a spectrally modeled sky.

## Situation

Atmospheric light changes with wavelength. A convincing sky renderer must account for effects such as Rayleigh and Mie scattering, ozone absorption, and the finite size of the Sun. Polarization and rainbows add further interactions to represent.

## Task

Vulkanic brings these effects together in an interactive renderer. It models 17 wavelength bands from 380 to 780 nm, builds lookup tables for atmospheric and particle scattering, and traces sky and optional rain-volume contributions on the GPU.

## Action

The C++ application computes spectral lookup tables, creates the Vulkan resources, and dispatches a compute shader for each frame. The shader evaluates atmospheric scattering, the solar disk, optional rainbow scattering, and procedural stars. It accumulates results in XYZ, converts them to display RGB, and presents the image in a GLFW window. Keyboard and mouse controls let you move the camera and inspect linear or elliptical polarization with a rotatable analyzer.

## Result

The result is an interactive sky-rendering demo that exposes how spectral scattering, polarization, and rain-volume parameters shape the image. Build and run it using the steps below.

## Requirements

- Windows, CMake 3.30 or newer, Ninja, Clang 20 or newer with `clang-cl`, and a Vulkan SDK with `glslc`
- A Vulkan 1.4 capable GPU and driver

CMake downloads GLFW, GLM, Vulkan Memory Allocator-Hpp, and vk-bootstrap through FetchContent.
The app prints its selected Vulkan device at startup.

## Build

From PowerShell, run:

```powershell
./build.ps1
```

The script builds the Windows GPU executable at `build-windows/Vulkanic.exe`.

To build manually:

```powershell
cmake --preset windows
cmake --build --preset windows --parallel
```

## Run

```powershell
./build-windows/Vulkanic.exe
```

The window title reports FPS and frame time. Controls:

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
