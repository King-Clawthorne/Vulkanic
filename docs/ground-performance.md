# Ground-view performance

The direct camera view of the Lambertian ground evaluated eight downwelling
sky directions per pixel and repeated their geometry/density integration for
all 13 spectral bands. Looking down therefore did much more work than looking
at the sky, even with a one-order atmosphere.

For atmosphere order 1, `ground_lighting.glsl` now calculates those directions,
ray intersections and density samples once, sharing them across wavelengths.
It retains the same eight directions, midpoint locations, minimum four view
steps, direct sunlight, diffuse skylight and ground-to-camera attenuation.
Optical depth is integrated as two wavelength-independent components, then
converted to spectral depth at use. For unpolarized direct sunlight, diffuse
ground intensity uses the Mueller intensity coefficients directly, avoiding
unnecessary polarization rotations while preserving the same intensity. This change
does not replace lighting with a constant, reduce quality settings, or cache
irradiance across different ground positions.

For higher atmospheric orders the original per-band ground path remains: the
shared array increased register pressure in deep scattering kernels and was
slower locally. GPU pipeline specialization selects the path for the configured
order. Higher-order ground lighting still has a substantial physical integration
cost; this optimization specifically addresses the default one-order profile.

## Local validation

On the RTX 5060 Ti at 960x540, the latest matched ground-facing comparison
improved from 1.895 ms to 1.495 ms median GPU render time, a further 21% reduction.
The original implementation measured 6.661 ms in the preceding optimization run.
Both latest captures used 30 warm-up and 60 measured frames, atmosphere
view/continuation/direction counts and orders of 1, rain disabled and unchanged
default optical settings. The camera was at `[0,2,-10]`, looking toward
`[0,-10,-9.9]`.

The latest ground HDR maximum absolute RGB difference was `4.657e-10`;
comparison passed the existing `2e-6 + 2e-4 * abs(reference)` tolerance. All five
matched cases passed: ground, sky, mixed horizon, higher orders and black ground.
Sky, higher orders and black ground matched exactly. The mixed-horizon maximum
absolute difference was `9.313e-10`. Sky-only median GPU time changed from
0.418 ms to 0.377 ms; the higher-order fallback remained effectively unchanged
at 18.597 ms versus 18.523 ms. These GPU pass timings are local measurements,
not guaranteed application frame rates.

Configurations are retained in the ignored `cmake-build-ninja/ground-validation`
directory. Latest baseline executable, before/after HDR and logs are in
`cmake-build-ninja/ground-more`. Use `scripts/compare-hdr.py` with its before/after
captures to reproduce numerical comparisons. `scripts/build.ps1` and both CTest
tests passed; the UI and README PNG are unchanged.

## Combined attenuation follow-up

Camera attenuation now accumulates in the same two-component optical-depth
representation. Adding the camera, half-slab and sunlight depths before the
spectral exponential replaces separate slab exponentials, square roots and
13 persistent camera transmittances. Sample locations and counts are unchanged.

A matched RTX 5060 Ti run measured 1.480 ms before and 1.247 ms after at 960x540,
a further 16% reduction. A 16-view-step case at 320x180 improved from 0.793 ms
to 0.674 ms. Each used 30 warm-up and 60 measured frames. All six HDR comparisons
passed: ground, sky, mixed horizon, higher orders, black ground and 16 view steps.
Ground maximum absolute RGB error was `4.075e-10` in both ground step settings.
Sky, higher orders and black ground remained bit-identical. The unchanged
higher-order fallback measured 18.253 ms versus 18.990 ms in this run; this
change targets the one-order path. Build and both CTest tests passed.

Before/after executables, shader source snapshot, captures and logs for this
follow-up are retained under `cmake-build-ninja/ground-fused`.

## Constant sunlight factors

Sun radiance, solar solid angle and step length now multiply the completed
per-direction integral once, instead of multiplying each spectral sample.
The attenuation, phase matrices and quadrature are unchanged.

Two ground trials measured approximately 1.253 ms before and 1.054 ms after
on the RTX 5060 Ti at 960x540, a further 16% reduction. The confirmation trial
ran the candidate first to check ordering effects. Each capture used 60 warm-up
and 120 measured frames. At 320x180 with 16 view steps, median time improved
from 0.677 ms to 0.578 ms. Mixed horizon improved from 0.245 ms to 0.228 ms.

All six matched HDR checks passed. Ground maximum absolute RGB difference was
`4.657e-10`; the 16-step case was `2.911e-10`. Sky, higher atmospheric orders
and black ground remained bit-identical. Both CTest tests and the build passed.
Explicit loop unrolling and alternate temporary-array layouts did not improve
performance and were discarded. Snapshots, final logs and comparisons are in
`cmake-build-ninja/ground-unroll` (the experiment directory retains its initial
name). No quality settings or integration sample counts changed.

## Shared phase coordinates and depth constants

The optical-depth loop now applies constant step length and aerosol extinction
after summing density samples. Mie lookup coordinates are computed once per
hemisphere direction, with the same interpolation and intensity entries as
`mie_mueller`. Solar factors and phase/quadrature normalization are applied
once after summing all directions. No sample counts or positions changed.

On the RTX 5060 Ti at 960x540, ground median GPU time decreased from 1.054 ms
to 1.010 ms (4.2%). The final candidate repeated at 1.010 ms with reversed
baseline/candidate execution order. At 320x180, mixed horizon measured
0.228 ms before versus 0.225 ms after; 16 view steps measured 0.578 ms versus
0.570 ms. Higher atmospheric orders retained the original path and measured
18.421 ms versus 18.409 ms.

All six HDR checks passed with 60 warm-up and 120 measured frames per capture.
Ground maximum absolute RGB difference was `5.239e-10`; sky, higher orders and
black ground were bit-identical. Build, both CTest tests and diff checks passed.
Snapshots, logs and captures are retained in `cmake-build-ninja/ground-depth`.
