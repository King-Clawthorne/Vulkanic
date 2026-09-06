# Coupled air and rain transport

The renderer sums Rayleigh, aerosol and rain Mueller scattering at each
interaction point. Each successive order gathers the previous order through
that same combined medium, including attenuation by all species. This includes
droplet-to-droplet paths and both directions of air/rain coupling. Rain is no
longer composited onto an independently calculated sky.

## Independent rain quality controls

The rainbow owns `rainbow.viewSteps` and `rainbow.scatteringOrders`; neither
inherits changes to the atmosphere settings. Both are saved by F5 and editable
in the GUI. Missing keys use fixed defaults of 5 view steps and 1 order.

| Setting | Meaning |
| --- | --- |
| `sky.spectralConstants.SCATTERING_ORDERS` | Maximum air/ground source order, 1–4 |
| `rainbow.scatteringOrders` | Maximum rain source order, 1–4 |
| `sky.spectralConstants.VIEW_STEPS` | Primary integration steps outside the rain interval |
| `rainbow.viewSteps` | Primary integration steps inside the rain interval; also rain solar-depth integration, with a minimum of 8 for that integral |
| `sky.spectralConstants.Samples` | Continuation-ray integration steps for the coupled solver |
| `sky.spectralConstants.secondarySamples` | Incident-direction and solar-disk samples |

Light transport stays coupled. At order N, air and rain source terms are each
included only if their own order limit is at least N. They gather the combined
previous-order field, so mixed air/rain paths remain. Extinction always includes
both species, regardless of source-order limits. For example, air order 1 and
rain order 2 retain direct air scattering and second-order rain illuminated by
both first-order air and rain; second-order air is omitted.

The short rain interval is split from the longer atmospheric interval to avoid
missing local rain. Both species at a sample point share the interval's spatial
quadrature, so changing rain steps can also change the numerical accuracy of air
inside that interval. Independent controls do not imply physically independent
images.

Incident directions use a mixture of uniform sampling and importance sampling
from the calculated droplet phase CDF. The actual mixture PDF weights every
sample, including all spectral bands. Samples change deterministically with
pixel, frame and path; stationary-camera accumulation resolves the angular
integral over time. This avoids the artificial secondary arcs produced by a
small fixed set of incident directions. Low frame counts can still be noisy.

Each slab uses the exact constant-coefficient source integral
`(1 - exp(-extinction * distance)) / extinction`, with a stable zero-extinction
limit. The density and source are still sampled spatially: increasing the integration
counts is needed for numerical convergence. The exponential attenuation model
is described in [PBRT's transmittance chapter](https://pbr-book.org/4ed/Volume_Scattering/Transmittance).

Ground reflection advances the transport order, as does a volume interaction.
The former atmosphere-specific ground-light shortcut is not mixed into the
coupled orders. The rain slider preserves `extinction - scattering` when edited,
so moving it away and back restores the same absorption.

## Atmospheric refraction

`SEA_LEVEL_REFRACTIVITY` controls sea-level `n - 1`; zero gives straight rays.
`REFRACTION_SCALE_HEIGHT` is measured in metres. The model is
`n(h) = 1 + SEA_LEVEL_REFRACTIVITY * exp(-h / REFRACTION_SCALE_HEIGHT)`.

The CPU integrates the eikonal ray equation with RK4 and calculates optical
depth along those curved trajectories. The GPU interpolates the resulting
table for view rays, sunlight and paths between interactions. Solar directions
are found by shooting toward the specified direction outside the atmosphere.
Solar-disk quadrature includes the solid-angle Jacobian of the refractive ray
mapping. Polarization frames are parallel-transported along radial-atmosphere
rays, and radiance transport accounts for the `L / n²` invariant.

The ray table has separate ground and space branches around the refracted
horizon, avoiding interpolation across an occlusion discontinuity. It is rebuilt
only when radii, density scale heights or refractivity change. Camera movement,
scattering coefficients, integration counts and sun direction reuse it.

The geometric method follows the ray-equation approach discussed in
[Ihrke et al., Eikonal Rendering](https://resources.mpi-inf.mpg.de/EikonalRendering/pdf/sig07.pdf).
This implementation uses an achromatic, radially stratified atmosphere. It does
not simulate temperature inversions, humidity profiles, ducting or mirages;
configuration validation requires increasing `n(r) * r`. Table interpolation and
finite integration resolution remain numerical error sources.

## Spherical droplets

Water droplets are spherical. The CPU traces Snell refraction and polarized
Fresnel reflection/transmission, including external reflection, direct
transmission and primary/optional secondary internal-reflection families.
Geometry is reused across the size distribution. The phase table is normalized
to `integral(F11 dOmega) = 4 pi` and preserves the spherical Mueller identities.

Airy-scale Gaussian broadening approximates wave-optical smoothing; solar
broadening is calculated in the illumination integral. Finite ray sampling,
Debye-family truncation and normalization remain model limits. There is no
nonspherical shape control or orientation average.

## Validation and performance

```powershell
.\scripts\build.ps1
ctest --test-dir cmake-build-ninja --output-on-failure
python .\scripts\validate-transport.py
```

CPU checks cover Fresnel energy and total internal reflection, spherical bow
dispersion, spherical Mueller positivity, phase normalization, independent quality
controls, ray invariants, analytic vertical optical depth and zero refraction.
The GPU script checks finite HDR output, Vulkan validation logs, orders 1–4,
isolated rain-to-rain scattering, the zero-rain limit, refraction changes
and independent rain controls, plus refracted solar visibility below
the geometric horizon. It retains its inputs and results in
the ignored `cmake-build-ninja/transport-validation` directory.

Four-wavelength packets share geometric work while retaining all thirteen
spectral bands. For local comparisons, configure
`-DVULKANIC_SPECTRAL_PACKET_WIDTH=1`, `2` or `4`; these change work grouping, not
the requested samples or scattering order. The default is 4.

High orders multiply the cost of angular and spatial quadrature. Realtime
settings are intentionally coarse and are not convergence-quality results.
Higher-count README captures demonstrate the renderer at their documented
settings, not guaranteed realtime performance at every quality setting.

See the [validation report](transport-validation.md) for measured timings,
reproducible configurations and packet-comparison numerical differences.
