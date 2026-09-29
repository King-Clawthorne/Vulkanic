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

- WSL 2 with a Linux distribution
- CMake 3.30 or newer, Ninja, GCC 15 or newer, Git, and pkg-config installed in WSL 2
- Linux Vulkan development files, `glslc`, and oneTBB development files installed in WSL 2
- A Vulkan 1.4 capable GPU and driver

CMake downloads GLFW, GLM, Vulkan Memory Allocator-Hpp, and vk-bootstrap through FetchContent.
On Ubuntu, install build tools and headers with `sudo apt install cmake ninja-build g++ git pkg-config glslc libvulkan-dev libtbb-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libwayland-dev libxkbcommon-dev`.

## Build

From PowerShell, run:

```powershell
./build.ps1
```

The script checks that it is running under WSL 2, configures a Release build with the `wsl` CMake preset, and writes the executable to `build-wsl/Vulkanic`. Native Windows CMake builds are rejected.

To build manually:

```sh
cmake --preset wsl
cmake --build --preset wsl --parallel
```

## Run

```sh
./build-wsl/Vulkanic
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
