# Vulkanic

Vulkanic is a real-time Vulkan renderer for exploring a polarized atmospheric sky.

## Situation

Vulkanic models Rayleigh scattering, the finite size of the Sun, and polarization in three color channels.

## Task

Vulkanic traces atmospheric scattering on the GPU using fixed Rayleigh coefficients for the red, green, and blue channels. The coefficients approximate inverse fourth-power scattering at representative wavelengths of 650, 550, and 450 nm. Each channel uses deterministic composite Gauss-Legendre integration with 16 evaluations per primary view ray. A squared distance mapping concentrates evaluations near the ray origin. The solar source has equal intensity across all 3 color channels.

## Action

The C++ application computes a Rayleigh density table, creates the Vulkan resources, and dispatches a compute shader for each frame. The shader evaluates atmospheric scattering and the solar disk in each color channel, tone-maps the results, and presents the image in a GLFW window. Keyboard and mouse controls let you move the camera and inspect linear or elliptical polarization with a rotatable analyzer.

Startup preprocessing integrates Rayleigh density columns. This table contains an optical property, not rendered sky pixels. The compute shader still traces scattering and evaluates sunlight, polarization, and camera-dependent rays each frame.

## Result

The result is an interactive sky-rendering demo that exposes how atmospheric scattering and polarization shape the image. Build and run it using the steps below.

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
It also regenerates the Doxygen HTML documentation at `build/docs/html/index.html`. Doxygen must be installed and available on `PATH`.

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

The HTML documentation is written to `build/docs/html/index.html`. The build
script and CMake docs target copy the image assets into the HTML output. For a
direct Doxygen run, copy them with:

```powershell
cmake -E copy_directory assets build/docs/html/assets
```

Doxygen and Graphviz `dot` are required for documentation generation. The build
script and CMake target locate `dot` in the default Graphviz install directory.
For a direct Doxygen run, make sure `dot` is on `PATH`.

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

Measured on October 21, 2026, using the Release build with deterministic composite Gauss-Legendre integration and a stationary view.

| GPU | Driver | GPU workloads | Median | Mean | 95th percentile |
| --- | --- | ---: | ---: | ---: | ---: |
| NVIDIA GeForce RTX 5060 Ti (GB206) | 32.0.16.1742 | 1,201 | 0.214 ms | 0.216 ms | 0.219 ms |

Timings are per-frame Vulkan GPU workload durations from Nsight Systems 2026.3.2's [Vulkan workload trace](https://docs.nvidia.com/nsight-systems/UserGuide/), after excluding the first 120 workloads as warm-up. The 1,201 measured workloads span 8.32 seconds. The 95th percentile uses linear interpolation between sorted durations. Vulkanic submits one compute dispatch per frame. These timings cover the GPU workload for the submission, not only the shader, and are not end-to-end FPS.

To capture a run, build the app and launch it through the Nsight Systems CLI:

```powershell
$nsys = "$env:ProgramW6432\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe"
& $nsys profile --duration=10 --trace=vulkan --vulkan-gpu-workload=individual --sample=none --cpuctxsw=none --force-overwrite=true --output=build/vulkanic-benchmark ./build/Vulkanic.exe
```

This path matches Nsight Systems 2026.3.2 on the benchmark machine. Adjust the version or install path if needed. The capture lasts 10 seconds, then Nsight Systems finalizes `build/vulkanic-benchmark.nsys-rep` and terminates the launched app. Keep the camera and scene fixed during that interval. The reported values come from the GPU timestamps in Nsight Systems Vulkan Workload records (`VULKAN_WORKLOAD`). The Vulkan API summary reports CPU-side API time instead.

Export the capture to a SQLite database for SQL queries:

```powershell
& $nsys export --type sqlite --output=build/vulkanic-benchmark.sqlite build/vulkanic-benchmark.nsys-rep
```

The `.sqlite` file is a database and not a text SQL script. Query its `VULKAN_WORKLOAD` table and calculate each duration as `(end - start) / 1000000.0` milliseconds. The `--force-overwrite=true` option lets either command replace the previous capture or database.

## Project layout

- `src/` - Vulkan renderer, settings, and CPU-side Rayleigh density table
- `shaders/Main.comp` - Vulkan compute shader
- `build.ps1` - clean configure and build helper
- `CMakeLists.txt` - dependencies, shader compilation, and executable target

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
