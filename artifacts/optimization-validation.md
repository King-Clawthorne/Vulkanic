# Optimization validation — 2026-09-05

The shared rainbow ray march reduced the median of three per-run GPU medians
from **29.49 ms to 15.15 ms** on the RTX 5060 Ti at 960x540: approximately **49%
less GPU time** for the sky/rainbow pass. This is a local result for the saved
configuration, not a claim of twice the application FPS across scenes or GPUs.

## What changed

- The rainbow shader marches the rain geometry and sunlight density once for
  all thirteen spectral bands. Wavelength-dependent attenuation remains per
  band, and each band's Mueller lookup and polarization transforms run once
  per view ray. Sampling counts, spectral bands and scattering orders are preserved.
- Benchmark mode provides native Vulkan timestamps, fixed input/config, and
  deterministic linear-HDR captures. Normal interactive runs do not allocate
  timestamp queries or record the benchmark timestamp synchronization.
- Every Vulkanic build synchronizes runtime SPIR-V files, including when a
  shader-only change does not relink the executable.
- Earlier changes enabled optimized Release shaders and factored repeated
  arithmetic out of the CPU rainbow histogram deposition loop.

## Nsight evidence

The user supplied `vulkan-profile.nsys-rep` after successfully capturing with an
elevated Nsight Systems 2026.3.2 CLI. Its SQLite export attributes 4,081 GPU
submission records to the **NVIDIA GeForce RTX 5060 Ti**. These records total
27.91 seconds. `vkWaitForFences` accounts for 26.10 seconds and 96.1% of recorded
Vulkan API duration; this points to GPU work as the performance target.

The supplied report contains submission-level GPU timing, not separate compute
dispatch timings. Submission durations vary strongly throughout the capture,
so it is not used as the baseline for a percentage improvement.

The earlier registry registration error no longer appeared on a non-elevated
retry, but `nsys profile` exited with `-1073741819` before producing a new
capture. The before/after measurements below use native Vulkan timestamps and
do not depend on another Nsight capture.

## Controlled GPU comparison

Both variants used the same benchmark executable and byte-identical saved
configuration, with Release `-O` shaders. The only shader difference is the
shared rainbow integration. Each trial ran 120 warm-up frames and 120 measured
frames; camera input, config reload and the GUI were disabled. Trials ran
sequentially in the order shown, never concurrently.

| Trial | Shader | Sky/rainbow mean (ms) | Sky/rainbow median (ms) |
| --- | --- | ---: | ---: |
| 1 | Before | 29.290477 | 31.434880 |
| 2 | After | 12.125313 | 10.869552 |
| 3 | After | 14.585636 | 15.151808 |
| 4 | Before | 28.305085 | 29.491888 |
| 5 | Before | 23.525388 | 19.372128 |
| 6 | After | 15.815460 | 16.096240 |

All three candidate medians were below all three baseline medians, though
run-to-run variation remains substantial. GPU clocks and other system workloads
were not controlled. The display pass median was approximately 0.02 ms. The
benchmark adds timestamp synchronization at pass boundaries, so these timings
should not be interpreted as uninstrumented end-to-end frame times.

The saved config uses 960x540, one sample per pixel, three sky scattering
orders, one primary/secondary sky integration step and direction, and 24 rain
view steps. It lives at `cmake-build-ninja/optimization-check/gpu-before/config.json`.

## Correctness and build checks

- MSVC C++26 Release build passed; both compute modules passed
  `spirv-val --target-env vulkan1.2`.
- An incremental `Vulkanic` target build copied runtime shaders without
  relinking the executable. Runtime and generated shader SHA-256 hashes matched.
- Eight before/after HDR comparisons passed: the full-resolution default view;
  320x180 views toward the bow, with rain disabled, outside a dense hard-edged
  rain volume, missing the volume, and with sunlight blocked by Earth; plus
  linear and elliptical analyzer variants of the bow view.
- Largest absolute linear-RGB difference across these comparisons:
  **1.1175870895385742e-8**. All values were finite and accumulation counts
  matched. Default captures contained 180 accumulated frames; other cases
  contained five. The three-frame-slot, rain-disabled case also exercised
  collection of pending timestamp results at shutdown.
- Analyzer tests used temporary shader copies with fixed analyzer vectors
  `(0, 1, 0)` and `(0.4330127019, 0.75, 0.5)` in both implementations. Production
  shader controls were not replaced with these constants.
- Tone-mapped HDR previews of the bow and dense-volume cases were visually
  inspected. Their appearance matched; this is capture-based visual checking,
  not a full interactive-control test. [Comparison image](optimization-comparison.png)
- Invalid benchmark frame counts are rejected before renderer startup.

The earlier CPU baker comparison checked all four components of all 53,261
table entries in seven runs (default repeats, primary-only, zero variance,
small droplets with variance 0.2, and large droplets with zero solar radius).
Maximum absolute difference was zero. Its default means were 3600.10 ms before
and 3496.81 ms after, with too much variation to establish a stable CPU speedup.
The earlier `-O0` versus `-O` window-title FPS tests were also inconclusive.

## Reproduction and artifacts

See the README's Benchmarking section for CLI and HDR format details.
`scripts/compare-hdr.py` uses only the Python standard library and exits nonzero
on invalid data, unmatched accumulation counts, or excessive RGB error.

Raw logs, frozen shader modules, config fixtures, HDR captures and temporary
analyzer shader sources are under `cmake-build-ninja/optimization-check` (ignored
build artifacts). In particular, `gpu-before/repeat-*.log` and
`gpu-after/repeat-*.log` hold the six timing trials. Retain those baseline
artifacts to rerun the comparison. The user's config edits, Nsight capture,
and existing `imgui.ini` were preserved.
