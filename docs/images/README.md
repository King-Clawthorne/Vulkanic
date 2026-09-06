# README renderer capture

`rainbow.png` is a 960 x 540 Vulkanic render of the coupled air/rain transport,
with atmospheric refraction, spherical droplets, and two scattering orders.
The sun is about 0.47 degrees above the geometric horizon, giving a warm
dawn/dusk background. The image contains 128 accumulated frames. It is not generated artwork or an upscaled
preview. The conversion applies the renderer's hue-preserving tone map and
linear-to-sRGB transfer at the configured exposure, without extra grading.

The complete input is committed as [rainbow-config.json](rainbow-config.json).
It is a showcase scene, separate from the editable default config.

| Setting | Capture value |
| --- | --- |
| Camera position / look-at | `[0, 2, -10]` / `[-0.35, 2.16, -10.25]` |
| Vertical field of view | 65 degrees |
| Exposure | 100 |
| Atmosphere / rain primary view steps | 4 / 4 |
| Shared continuation steps (`Samples`) | 2 |
| Incident-direction / solar-disk samples (`secondarySamples`) | 4 |
| Atmosphere / rain scattering orders | 2 / 2 |
| Samples per pixel per frame | 1 |
| Warm-up / measured frames | 16 / 112 |
| Analyzer | Off |
| Sun direction | `[0.35, 0.0035, 0.25]` |
| Aerosol extinction (`BETA_M`) | `0.000006` per metre |
| Sea-level refractivity / scale height | `0.000277` / 8000 metres |
| Rain centre / radii | `[-2500, 1800, -1800]` / `[3000, 3000, 3000]` metres |
| Rain scattering / extinction | Both `0.00012` per metre |
| Effective droplet radius / variance | 1000 micrometres / 0.08 |
| Angular table / secondary bow | 4097 bins / enabled |

Reproduce from the repository root after building:

```powershell
New-Item -ItemType Directory -Force cmake-build-ninja/readme-captures
.\cmake-build-ninja\Vulkanic.exe --benchmark 112 --warmup 16 --config .\docs\images\rainbow-config.json --capture-hdr .\cmake-build-ninja\readme-captures\rainbow.hdrbin
python .\scripts\preview-hdr.py .\cmake-build-ninja\readme-captures\rainbow.hdrbin .\docs\images\rainbow.png --exposure 100
```

The PNG converter requires Pillow. The capture used the Release build on an
RTX 5060 Ti; raw HDR and logs from this run are retained under the ignored
`cmake-build-ninja/transport-validation/readme-dusk.*` paths. This higher-count
capture took about 814 ms per transport frame, not realtime. Realtime timings
use the lower-count profile documented in the [validation report](../transport-validation.md).

See [transport model and limitations](../transport.md) for the finite scattering orders, numerical integration and wave-optics limits.
