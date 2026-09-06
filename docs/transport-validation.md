# Transport validation

Historical packet-comparison timings below were measured before removing nonspherical droplets;
rerun the supplied profile to measure the spherical model.

Measured locally on an NVIDIA RTX 5060 Ti using the Windows Release build.
These results describe the implemented geometric-optics model, not agreement
with measured rainbows or an exact electromagnetic reference.

## Correctness checks

- `ctest --test-dir cmake-build-ninja --output-on-failure` passes the CPU optics
  suite: Fresnel energy, total internal reflection, spherical bow dispersion,
  Mueller positivity, phase normalization/CDF, independent controls, absorption
  preservation, refracted-ray invariants and analytic reference trajectories.
- `python scripts/validate-transport.py` passes all 16 GPU cases. These cover
  orders 1–4, isolated rain-to-rain scattering, finite HDR output, zero-rain
  equivalence, independent rain controls, and refraction.
  Zero-rain and disabled-rain control comparisons are bitwise identical.
  Isolated-rain atmospheric-order independence is checked within floating-point
  tolerance (`1e-9 + 1e-6 * abs(reference)`). A separate
  horizon case shows sunlight below the geometric horizon becoming visible
  with refraction. No Vulkan validation errors were reported in these runs.
- Interactive F5 saving and config hot reload pass in an isolated temporary
  copy. Rain view steps/order survive saving and hot reload independently of
  the unchanged atmosphere controls. Refraction hot reload also remains active.
  This does not constitute a mouse-drag interaction test.

The GPU script retains configurations, logs, raw HDR and `results.json` under
the ignored `cmake-build-ninja/transport-validation` directory.

## Realtime profile and packet comparison

The complete measured input is [realtime-config.json](realtime-config.json):
960 × 540, one sample per pixel per frame, two scattering orders for both media,
`VIEW_STEPS = Samples = secondarySamples = 1`, rain enabled,
and atmospheric refraction enabled. All thirteen wavelength bands are retained.

Each trial used 60 warm-up frames and 120 measured frames. Trials ran sequentially
in packet-width order 4, 1, 1, 4. GPU transport timings were:

| Packet width | Trial | Median ms | Mean ms | 95th percentile ms |
| --- | --- | --- | --- | --- |
| 4 (default) | 1 | 8.332 | 8.382 | 8.581 |
| 1 (reference grouping) | 2 | 17.884 | 17.699 | 18.199 |
| 1 (reference grouping) | 3 | 15.506 | 16.396 | 18.101 |
| 4 (default) | 4 | 8.345 | 8.381 | 8.584 |

The display pass took about 0.007 ms in the default profile. These are GPU pass
timings, not guaranteed application frame rates. Low integration counts and
temporal accumulation are essential to this profile; higher orders and counts
can exceed the realtime budget substantially.

The two width-4 captures are bitwise identical. Comparing the 180-frame width-1
and width-4 HDR captures gives maximum absolute component difference
`2.15988e-5`, mean absolute difference `5.58680e-9`, and relative L2 difference
`2.05979e-5` (about 0.00206%). The strict comparison defaults (`atol=2e-6`,
`rtol=2e-4`) fail for 42 of 1,555,200 RGB components. An explicitly looser
`atol=3e-5` passes. Packet grouping therefore has a small measured numerical
difference and is not claimed to be bitwise equivalent to the scalar grouping.

Reproduce the default profile after building:

```powershell
.\cmake-build-ninja\Vulkanic.exe --benchmark 120 --warmup 60 --config .\docs\realtime-config.json --capture-hdr .\cmake-build-ninja\realtime.hdrbin
```

Configure separate CMake build directories with
`-DVULKANIC_SPECTRAL_PACKET_WIDTH=1` and `=4` to compare work grouping. Run the
same command/config from each executable, then use `scripts/compare-hdr.py` on
the two captures. CPU/GPU clock variation affects timing; retain repeated runs.

## README capture

The [README image](images/rainbow.png) uses 128 accumulated frames and higher
integration counts. Its median transport time was 814.403 ms per frame; it is
a quality capture, not the realtime benchmark. The exact configuration and
capture commands are in [images/README.md](images/README.md).

See [transport.md](transport.md) for finite scattering
orders, cached ray interpolation and approximate wave-optical smoothing limits.
