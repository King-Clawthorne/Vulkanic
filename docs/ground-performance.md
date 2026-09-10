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
