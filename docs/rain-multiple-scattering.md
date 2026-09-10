# Rainbow multiple scattering

The rainbow includes rain-to-rain paths through up to four scattering events.
Order 1 retains the original deterministic integrator. Orders 2–4 use backward
sampled paths and the existing polarized droplet Mueller matrix. Each event
gathers attenuated sunlight, then transports its Stokes vector through the
preceding droplets. The added terms are scattering orders, not brightness boosts.

Independent settings under `rainbow`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `scatteringOrders` | 2 | Maximum number of rain scattering events, 1–4 |
| `multipleScatteringSamples` | 1 | Paths per pixel sample per frame, 1–64 |
| `multipleScatteringSteps` | 8 | Optical-depth integration steps per segment, 1–64 |

`viewSteps` still controls the original first-order rainbow. Atmosphere controls
remain independent. The GUI layout is unchanged; use JSON for the new settings.
F5 serialization and hot reload include these fields. The rain enable flag and
fixed-absorption slider fix are retained.

Geometry and samples are shared across all 13 wavelengths. Distances use a
truncated-exponential proposal; directions use a mixture of the 550 nm phase
CDF and uniform sampling. Actual proposal densities weight every contribution.
Samples vary with pixel/frame and accumulate in the existing HDR history.
More paths or accumulated frames reduce noise. Work is bounded by the path
count and maximum order, rather than recursively multiplying quadrature counts.

## Limits

The spherical droplet model is retained. There is no atmospheric refraction or
mixed air-to-rain/rain-to-air scattering. Atmospheric attenuation of sunlight
is retained, but rain-path extinction uses rain coefficients, as in the original
rainbow. This is not a unified air/rain radiative-transfer solution.

The existing phase table keeps its primary/secondary Debye-family truncation,
Airy-scale smoothing and baked solar-disk broadening. Reusing it at each event
also repeats that broadening; this is not an exact full-wave rain model.
Finite optical-depth quadrature and finite order introduce bias, and sampled
paths introduce noise. Added orders contribute positive physical intensity,
but need not brighten the visible bow uniformly.

## Checks

```powershell
.\scripts\build.ps1
ctest --test-dir cmake-build-ninja --output-on-failure
python scripts/validate-rain-multiple.py
```

CPU tests cover the angular CDF against an isotropic reference, preservation of
phase entries, config validation and the slider round trip. GPU checks cover
finite HDR, increasing orders, zero rain and approximately quadratic second-order
density scaling. With `--baseline <old executable>`, the GPU script also requires
bit-identical order-1 output. Captures and logs are retained under the ignored
`cmake-build-ninja/rain-multiple-validation` directory.

The README PNG remains the original single-scattering capture.

## Local timing

On an RTX 5060 Ti at 960×540, with rain enabled and atmosphere primary steps,
continuation steps, direction samples and scattering orders all set to 1:

| Rain order | Median GPU render ms | 95th percentile ms |
| --- | --- | --- |
| 1 | 2.460 | 2.969 |
| 2 | 2.857 | 3.387 |

Both runs used 60 warm-up and 120 measured frames; rain view steps were 24,
multiple-scattering paths 1 and optical-depth steps 8. The camera looked toward
`[-0.35, 2.16, -10.25]` at 65 degrees FOV. These are GPU pass timings, not a
guaranteed application frame rate; higher quality settings cost more. Exact
configs and logs are saved as `realtime-1.*` and `realtime-2.*` in the validation
directory. The original GUI's F5 save and JSON hot reload were also exercised
with distinct rain and atmosphere orders.
