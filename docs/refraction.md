# Atmospheric refraction

This addition starts from the exact file tree of `5c795917b001a31eab7741477c7079758bd4011e`.
The original GUI, spherical droplet table, single-scattering rainbow and
separate atmosphere/rain integration controls are retained.

JSON settings under `sky.spectralConstants`:

- `SEA_LEVEL_REFRACTIVITY`: sea-level `n - 1`, default `0.000277`; zero selects
  the original straight-ray renderer.
- `REFRACTION_SCALE_HEIGHT`: exponential refractivity scale height in metres,
  default `8000`.

Both settings hot-reload and survive F5 saving. They have no GUI controls.
The model is `n(h) = 1 + N0 exp(-h/H)`. The CPU integrates the eikonal ray
equation using RK4 and caches trajectories and air optical depths. The GPU
interpolates separate ground/space branches, shoots local sunlight toward the
external sun direction, and transports polarization frames and the `L/n²`
radiance invariant along curved paths. Atmosphere orders 1–4 and the original
single-scattering rainbow use these trajectories. Tables rebuild only when
the atmosphere geometry or refractive profile changes.

This is an achromatic, radially stratified atmosphere. It excludes inversions,
trapped rays and weather-dependent profiles. Finite RK4 resolution, ray-table
interpolation and the original finite scattering quadrature introduce numerical
error. Local rain-volume intersections use a straight-ray seed padded by 250 m,
then evaluate density along the curved ray; grazing-volume intersections remain
an approximation. The original droplet and solar-broadening model is retained.

## Validation

```powershell
.\scripts\build.ps1
ctest --test-dir cmake-build-ninja --output-on-failure
python scripts/validate-refraction.py
```

CPU tests cover conserved ray impact parameter, unit tangents, analytic vertical
optical depth, zero-refraction trajectories and config validation. GPU checks
cover finite HDR, all four atmospheric orders and refracted solar visibility
below the geometric horizon. A local build of `5c79591` produced bitwise-identical
HDR to this build with refractivity zero in the regression scene. F5 saving and
refraction hot reload pass with the unchanged original GUI implementation.

The rollback restores the pre-existing `VUID-vkQueueSubmit-pSignalSemaphores-00067`
presentation warning. It was reproduced in both the unmodified baseline and this
build. It is not repaired by this refraction-only change. The validation script
fails on validation errors by default; `--allow-existing-present-warning`
explicitly records and permits only this known warning. Other failures still
fail the run. Results and logs remain in `cmake-build-ninja/refraction-validation`.

The [README capture](images/README.md) uses the original saved scene configuration
with only the two refraction fields added. Exposure, camera, lighting, rain
parameters, resolution and accumulated frame count are unchanged. High-quality
capture settings are not a realtime performance claim.

## Measured cost

On the local RTX 5060 Ti, the default 960×540 configuration with VSync disabled
measured 41.448 ms median GPU transport time over 120 frames after 60 warm-up
frames. The original README scene with refraction measured 327.733 ms median
at its higher quality settings. The current refracted path retains the original
per-wavelength integrators and costs more than the straight-ray fast path;
these measurements do not establish 60 FPS performance. Inputs and logs are
retained in the local validation directory.
