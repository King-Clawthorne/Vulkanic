# Further quality-preserving optimization — 2026-09-06

Two changes reduce redundant work while retaining the existing rendering model:

1. **Apply the single-scattering phase transform once per ray integral.**
   Along a straight ray, incident sunlight and viewing directions are fixed.
   The Rayleigh and Mie matrix contributions are linear in their respective
   density/source weights. Accumulate those two weights at the original
   integration points, then apply the Mueller transform and final physical
   clamp once. This also benefits the single-scattering evaluations nested
   inside ground irradiance and higher scattering orders. Intermediate
   polarization throughout multiple scattering is retained.
2. **Cache spectral coefficients on config upload.** Compute the same thirteen
   Rayleigh coefficients and normalized Planck solar radiances on the CPU,
   using the existing wavelengths, float formulas and constants. Upload them
   in a 336-byte scene UBO instead of recomputing them throughout GPU transport.
   Config reload recomputes the cache. No table-resolution or spectral-band
   reduction is involved.

Resolution, integration counts, scattering orders, polarization modes, rain
features and the default JSON are unchanged. Arithmetic reassociation and
CPU/GPU elementary-function rounding introduce small numerical differences;
this is not a bit-identical transformation.

## Measured performance and uncertainty

Baseline is commit `065017f`, including the previous specialization and
filter-off optimizations. The baseline runtime is preserved in the ignored
`cmake-build-ninja/fps-round2/baseline` directory. Measurements use native Vulkan
sky/rainbow-pass timestamps on the RTX 5060 Ti at the default 960x540 settings.

The initial quieter tests (120 warm-up and 180 measured frames) recorded:

| Version | Mean ms | Median ms |
| --- | ---: | ---: |
| Baseline | 7.297097 | 7.262784 |
| Shared phase transform | 6.605978 | 6.480368 |
| Shared phase + cached spectrum | 6.080164 | 5.947456 |

Later runs became substantially noisier, including the unchanged display pass.
The final sequential baseline/candidate pairs used 120 warm-up and 240 measured
frames each. All three pairs are reported, including the unfavorable first pair:

| Pair | Version | Mean ms | Median ms | p95 ms |
| --- | --- | ---: | ---: | ---: |
| 1 | before | 10.471248 | 11.473568 | 14.721696 |
| 1 | after | 11.972125 | 12.490320 | 14.944864 |
| 2 | before | 11.864220 | 11.667712 | 17.654976 |
| 2 | after | 7.644880 | 6.355856 | 12.755360 |
| 3 | before | 8.953673 | 7.616240 | 13.442784 |
| 3 | after | 7.040837 | 6.241680 | 11.349152 |

The relatively quiet third pair gives **18.0% less GPU time**, or about 22%
higher throughput for the measured pass. The initial quiet measurements are
consistent with that scale of improvement. However, the later runs do not
support a fixed, repeatable FPS percentage under uncontrolled load. GPU clocks
and other workloads were not locked. These are pass timings, not end-to-end
application FPS. Do not aggregate the noisy pairs into a large speedup claim.

## Experiments not retained

- Specializing the rain step count: 6.477 ms versus 6.480 ms for the preceding
  candidate; negligible in the initial default-scene test.
- Hoisting rain-coordinate scaling: 6.430 ms versus 6.480 ms; under 1% at default.
  A small bow test also showed only a small absolute change (~0.01 ms).
- Compile-time-known CIE wavelength branches: no reliable gain over the existing
  colour fit; a nearby reference rerun was effectively tied.
- CPU-cached CIE weights: HDR passed, but timing became noisy and showed no
  reliable additional gain. The extra uniform data and duplicate formulas
  were removed.
- 32x2 workgroups: HDR passed, but performed worse in the observed runs under
  variable load. Retained the existing 8x8 layout and dispatch.

Stopped after these remaining tested approaches were negligible, slower or
inconclusive. This does not establish that no future architectural optimization
is possible. No quality-reducing feature or sampling change was substituted.

## Validation

Fifteen deterministic baseline/candidate HDR comparisons passed the existing
`compare-hdr.py` thresholds: absolute 2e-6 plus relative 2e-4 per RGB component,
with exact accumulation-count matching and finite-value checks:

- Default 960x540 view.
- Bow, Earth shadow, missed rain volume, hard-edged rain and rain disabled.
- Orders 1–4, each with primary/secondary integration and direction counts of
  two, at 161x91 to cover partial edge workgroups.
- 3500 K and 12000 K solar spectra with larger integration counts.
- A view into the solar disk.
- Linear and elliptical analyzer variants of the hotter-sun scene.

The solar-disk capture has the largest absolute HDR difference, 0.00061035,
because it contains bright radiometric values; its relative L2 difference is
3.15e-7. Other compared cases have maximum absolute error no greater than
5.96e-8. Relative L2 errors remain below 4.3e-7 throughout the suite. The
side-by-side tone-mapped bow previews were visually inspected and matched.
Polarizer tests used temporary shader copies with identical fixed analyzer
vectors in both implementations; production controls were unchanged.

A separate hidden-window smoke test changed solar temperature and Rayleigh
strength twice. Both config reloads were logged, stderr was empty, and the
renderer closed normally with exit code zero. This verifies the reload path;
it is not an exhaustive interactive GUI test.

Release build, SPIR-V validation for Vulkan 1.2 and `git diff --check` passed.
The generated and runtime path-tracer SPIR-V hashes matched. Raw logs, temporary
configs, variant binaries, test scripts and HDR captures are retained under
`cmake-build-ninja/fps-round2`. Example local reproduction:

```powershell
python cmake-build-ninja/fps-round2/bench.py baseline repeat-before config/path_tracer_config.json 240 120
python cmake-build-ninja/fps-round2/bench.py candidate repeat-after config/path_tracer_config.json 240 120
python scripts/compare-hdr.py cmake-build-ninja/fps-round2/repeat-before.hdr cmake-build-ninja/fps-round2/repeat-after.hdr
```
