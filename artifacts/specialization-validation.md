# Shader specialization validation — 2026-09-06

The default sky/rainbow pass took **10.7% less GPU time** on the local RTX 5060 Ti:
median-of-run-medians fell from 8.171 ms to 7.298 ms. This corresponds to about
12% higher throughput for that pass, not measured end-to-end application FPS.

## Implementation

Vulkan specialization constants supply the existing scattering order, primary
view steps, secondary integration steps and secondary direction count. The
compiler can eliminate unused orders and optimize constant loop bounds.
No resolution, sample count, wavelength, physics formula or default config changed.
The shader retains its uniform-driven path when specialization values are zero.
Changing these four settings waits for in-flight work and recreates the pipelines;
this can introduce a brief compilation pause when editing quality settings.

## Paired timing

Baseline: clean commit `7c645c0`, frozen executable/shaders in
`cmake-build-ninja/fps-check/before`. Candidate: this working tree, Release build.
Both use the unchanged `config/path_tracer_config.json`, 960x540, 120 warm-up
frames and 240 measured frames. Runs were sequential, in the following order.

| Variant | Mean GPU ms | Median GPU ms |
| --- | ---: | ---: |
| Specialized | 7.406938 | 7.297536 |
| Baseline | 8.184754 | 8.183088 |
| Baseline | 8.156087 | 8.158048 |
| Specialized | 7.423393 | 7.297888 |

Display-pass medians were 0.0071–0.0072 ms. Measurements use native Vulkan
GPU timestamps; clocks and other system workloads were not controlled.
Workgroup-size and loop-unroll-hint experiments were inconclusive and removed.

## Quality checks

Ten deterministic HDR comparisons passed `scripts/compare-hdr.py` with its
unchanged tolerances (absolute 2e-6, relative 2e-4): default view, bow view,
Earth shadow, missed rain volume, hard-edged rain, rain disabled, and orders
1–4 with all three sky integration/direction counts set to two. The order
cases use 161x91 to exercise partial edge workgroups. Counts matched and
all output components were finite. Maximum absolute RGB difference across
these captures was 8.5682e-7; the four order cases were bit-identical.
Side-by-side tone-mapped bow captures were visually inspected and matched.
These checks do not establish bit-identical output across all scenes or GPUs.

MSVC Release build and SPIR-V validation for Vulkan 1.2 passed. A hidden-window
config-reload smoke run stayed alive through order changes, but buffered logs
could not confirm each reload; interactive hot-reload acceptance remains unverified. Raw timing logs,
HDR captures, temporary case configs and comparisons are retained in the ignored
`cmake-build-ninja/fps-check` directory. Reproduce timing with each executable:

```powershell
Vulkanic.exe --benchmark 240 --warmup 120 --config <absolute-config-path>
```

For matching quality captures, use identical frame counts in both variants and
add `--capture-hdr <absolute-output-path>`, then run:

```powershell
python scripts/compare-hdr.py before.hdr after.hdr
```
