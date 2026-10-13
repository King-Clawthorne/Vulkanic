# Vulkanic

Vulkanic is a real-time Vulkan path tracer for exploring a spectrally modeled sky.

## Situation

Atmospheric light changes with wavelength. A convincing sky renderer must account for effects such as Rayleigh and Mie scattering, ozone absorption, and the finite size of the Sun. Polarization and rainbows add further interactions to represent.

## Task

Vulkanic brings these effects together in an interactive renderer. It samples one random wavelength from 380 to 780 nm for each path sample, interpolating lookup tables built at 17 reference wavelengths. It traces sky and optional rain-volume contributions on the GPU.

## Action

The C++ application computes spectral lookup tables, creates the Vulkan resources, and dispatches a compute shader for each frame. The shader evaluates atmospheric scattering, the solar disk, and optional rainbow scattering at the sampled wavelength. It accumulates results in XYZ, converts them to display RGB, and presents the image in a GLFW window. Keyboard and mouse controls let you move the camera and inspect linear or elliptical polarization with a rotatable analyzer.

Startup preprocessing integrates atmospheric density columns and computes wavelength-dependent particle phase functions at higher resolution. These tables contain optical properties, not rendered sky pixels. The compute shader still traces scattering and evaluates sunlight, rain, polarization, and camera-dependent rays each frame.

## Result

The result is an interactive sky-rendering demo that exposes how spectral scattering, polarization, and rain-volume parameters shape the image. Build and run it using the steps below.

![Vulkanic renderer output showing the solar disk and atmospheric scattering](assets/renderer-result.png)

## Requirements

- Windows, CMake 4.4 or newer, Ninja, Clang 20 or newer with `clang-cl`, and a Vulkan SDK with `glslc`
- A Vulkan 1.4 capable GPU and driver

CMake downloads GLFW, GLM, Vulkan Memory Allocator-Hpp, and vk-bootstrap through FetchContent.
The app prints its selected Vulkan device at startup.

## Build

From PowerShell, run:

```powershell
./build.ps1
```

The script builds the Windows GPU executable at `build/Vulkanic.exe`.
It also regenerates the Doxygen HTML documentation at `build/docs/html/index.html`; Doxygen must be installed and available on `PATH`.

CMake's CTest integration can create a `build/Testing` directory when tests
are run. This project does not currently define tests, so that directory is
only CTest's run log and is safe to remove while no CTest process is running.

To build manually:

```powershell
cmake --preset windows
cmake --build --preset windows --parallel
```

## Documentation

For manual builds, generate the C++ and shader reference with either command:

```powershell
doxygen Doxyfile
```

```powershell
cmake --build --preset windows --target docs
```

The HTML documentation is written to `build/docs/html/index.html`.

## Run

```powershell
./build/Vulkanic.exe
```

The window title reports FPS and frame time. Controls:

- **W/A/S/D**: move forward, left, backward, and right
- **Space / Left Ctrl**: move up and down
- **Right mouse button + mouse**: look around
- **[ / ]**: rotate the polarizer
- **P**: toggle the polarizer
- **C**: toggle elliptical polarization display
- **Escape**: close the window

## GPU performance profiling

Measured on October 13, 2026, using the Release build and default 1280 × 720 scene settings. The view remained stationary with the default camera. Rain rendering was disabled.

| GPU | Driver | GPU workloads | Median | Mean | 95th percentile |
| --- | --- | ---: | ---: | ---: | ---: |
| NVIDIA GeForce RTX 5060 Ti (GB206) | 32.0.16.1742 | 17,838 | 0.843 ms | 0.869 ms | 1.039 ms |

Timings are per-frame Vulkan GPU workload durations from Nsight Systems 2026.3.2's [Vulkan workload trace](https://docs.nvidia.com/nsight-systems/UserGuide/), after excluding the first 120 workloads as warm-up. The 17,838 measured workloads span 17.3 seconds. Vulkanic submits one compute dispatch per frame. These timings cover the GPU workload for the submission, not only the shader, and are not end-to-end FPS.

To capture a run, build the app and launch it through the Nsight Systems CLI:

```powershell
$nsys = "$env:ProgramW6432\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe"; & $nsys profile --duration=20 --trace=vulkan --vulkan-gpu-workload=individual --sample=none --cpuctxsw=none --output=build/vulkanic-benchmark ./build/Vulkanic.exe
```

This path matches Nsight Systems 2026.3.2 on the benchmark machine. Adjust the version or install path if needed. The capture lasts 20 seconds, then Nsight Systems finalizes `build/vulkanic-benchmark.nsys-rep` and terminates the launched app. Keep the camera and scene fixed during that interval. The reported values come from the GPU timestamps in Nsight Systems Vulkan Workload records (`VULKAN_WORKLOAD`). The Vulkan API summary reports CPU-side API time instead.

## Project layout

- `src/` - Vulkan renderer and CPU-side sky/scattering tables
- `shaders/Main.comp` - Vulkan compute shader
- `build.ps1` - clean configure and build helper
- `CMakeLists.txt` - dependencies, shader compilation, and executable target

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
