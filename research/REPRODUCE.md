# Reproducing the experiments

Run commands from the `research` directory in PowerShell. This directory contains the shader, harness, build configuration, scripts, and data needed for the experiment, independently of the renderer in the repository root. The recorded environment was Windows with CMake 4.4.4, Clang 22.1.8 (`clang-cl`), Ninja, Vulkan SDK `glslc` from shaderc 2026.3, Python 3.13.5, NumPy 2.1.3, Matplotlib 3.10.0, Pandoc 2.12, MiKTeX XeLaTeX 4.18, Nsight Systems 2026.3.2, and an NVIDIA GeForce RTX 5060 Ti using driver 32.0.16.1742. GPU timings depend on the device and driver.

## Build and CPU experiments

```powershell
$cmake = 'C:\Program Files\CMake\bin\cmake.exe'
python -m pip install numpy==2.1.3 pandas
& $cmake -S . -B build\local -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang-cl
& $cmake --build build\local --target PolarizedResearchHarness --parallel
```

The CMake path above is the recorded installation. Change it to the CMake 4.4 or newer executable if it is installed elsewhere. The standalone build uses Ninja and `clang-cl` and needs the Vulkan SDK with `glslc` and the Vulkan Memory Allocator header at `vma/vk_mem_alloc.h`. CMake also needs Git and network access on the first configuration to fetch vk-bootstrap v1.4.357 at commit `556b79b165386f6c1a18362d30f2a076fdaa2778`. Install NumPy and pandas before running the validation commands.

`results/cpu_reference_stokes.csv` retains the double-precision CPU reference values for the 20 recorded geometries. `results/cpu_reference_metadata.json` records their settings, environment, and input hashes. The CPU reference code has been removed, so these values cannot be regenerated or extended to new geometries from this artifact. The validation script reads the retained values directly.

## GPU sweep and Nsight timing

```powershell
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe'
& $nsys profile --trace=vulkan --force-overwrite=true --output build\polarized-research-gpu build\local\PolarizedResearchHarness.exe results\gpu_sweep.csv
& $nsys export --type=sqlite --force-overwrite=true --output build\polarized-research-gpu.sqlite build\polarized-research-gpu.nsys-rep
python validate.py gpu
```

The harness executes three warmup and 21 measured submissions for each of ten configurations. Nsight records the Vulkan workload duration for each submission. Each submission processes 320 rays, made from 16 replicas of 20 geometries. These are workload durations, not end-to-end frame times. The retained paper timings were measured before the harness switched to vk-bootstrap and Vulkan Memory Allocator. Queue or memory selection can differ on a new run, so remeasure timings rather than assuming they will match.

`gpu_sweep.csv` is the raw Vulkan output. The GPU validation command reads this file and the retained CPU reference CSV, then writes `gpu_validation.csv`, which adds reference Stokes components and relative intensity and Stokes errors. The script does not check for missing geometries or mismatched view panel counts. Verify the inputs match the reference settings in the metadata before interpreting comparisons. Analyze this file in a spreadsheet or another tool. For the reported cap counts, count adaptive rows with `max_selected_azimuth` at least 256, matching the original analysis.

For manual DoLP and AoLP calculations, each color has GPU columns such as `red_I`, `red_Q`, and `red_U`, and reference columns such as `cpu_red_I`, `cpu_red_Q`, and `cpu_red_U`. For each row:

1. For each color and for both GPU and reference values, calculate DoLP as `sqrt(Q*Q + U*U) / max(I, 1e-300)` and AoLP as `0.5 * atan2(U, Q)` in radians. Check the argument order in your spreadsheet's `atan2` function.
2. A color passes the intensity mask when its reference intensity is greater than `max(1e-6 * max(cpu_red_I, cpu_green_I, cpu_blue_I), 1e-300)`. The row's DoLP error is the maximum absolute GPU minus reference DoLP over those colors, or zero if none pass.
3. For AoLP, also require reference DoLP of at least 0.05. For each qualifying color, wrap the GPU minus reference angle into `[-pi/2, pi/2)` using `(difference + pi/2) mod pi - pi/2`, take its absolute value, and multiply by `180/pi`. The row's AoLP error is the maximum over qualifying colors, or undefined if none qualify.
4. To calculate held-out maxima, select adaptive rows with tolerance `0.0001` and split `heldout`, then take the maximum row error for each observable, excluding undefined AoLP values.

The retained `gpu_sweep_profiled.csv` and `gpu_timings.csv` contain the original run's timing data. To analyze a new profile, open the exported SQLite database in a SQLite viewer, inspect the `VULKAN_WORKLOAD` table, and sort rows by `start`. There should be 240 rows, in ten consecutive blocks of 24. The configuration order is point Sun, fixed azimuth counts 8, 16, 32, 64, 128, and 256, then adaptive tolerances 0.01, 0.001, and 0.0001 with a cap of 256. For each block, discard the first three warmups, convert each remaining `end - start` duration from nanoseconds to milliseconds by dividing by 1,000,000, and calculate the median, mean, and linearly interpolated 95th percentile of the 21 durations. Match each timing to its configuration when comparing it with the validation data.

## Independent timing repeat

The reported results include a second Nsight profile to check timing stability and deterministic outputs.

```powershell
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe'
& $nsys profile --trace=vulkan --force-overwrite=true --output build\polarized-research-gpu-repeat2 build\local\PolarizedResearchHarness.exe results\gpu_sweep_repeat2.csv
& $nsys export --type=sqlite --force-overwrite=true --output build\polarized-research-gpu-repeat2.sqlite build\polarized-research-gpu-repeat2.nsys-rep
```

Analyze the repeat's SQLite export in the same way, then compare each configuration's timing statistics with the first run. Compare corresponding Stokes components in `gpu_sweep.csv` and `gpu_sweep_repeat2.csv` to inspect output consistency. The retained `gpu_timings_repeat2.csv` and `gpu_sweep_profiled_repeat2.csv` contain the original repeat's timing data.

## Numerical conventions

- Retained CPU reference values were computed in double precision using 48 composite four-point Gauss-Legendre view panels and 96-point Gauss-Legendre optical-depth integration.
- GPU path: the same view panel count and nodes, with optical-depth integration split into composite four-point panels no longer than four scale heights.
- Retained source reference: gnomonic clipping of the solar disk at the planet horizon using order 64 Gaussian quadrature.
- The retained workflow does not include the cross-coordinate reference check, check camera-path convergence, or compare matching CPU and GPU source rules. The measured GPU errors therefore do not establish an end-to-end error bound.
- The relative Stokes error is the Euclidean norm over three RGB channels and four Stokes components, divided by the reference norm.
- DoLP error is an absolute difference. AoLP error is masked to reference intensity above 1e-6 of the maximum and reference DoLP of at least 0.05.
- For adaptive runs, a cap event means at least one camera-path node reached the maximum source azimuth count. The local indicator may or may not meet the tolerance at that count. The coarse-to-fine estimate is an empirical indicator, not a proven bound.

The retained raw and profiled GPU sweeps and timing tables preserve both measured runs. The validation command regenerates its CSV output. Polarization errors, summary tables, and figures require manual analysis. The retained figures use the CPU reference and are not included in the manuscript.

To rebuild the manuscript PDF:

```powershell
pandoc PAPER.md --pdf-engine=xelatex -V papersize:a4 -V fontsize=9pt -V geometry:margin=0.55in --output error-aware-polarized-skies.pdf
```

Build products and Nsight trace files remain under `research/build/`.
