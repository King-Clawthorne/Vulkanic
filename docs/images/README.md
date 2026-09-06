# README renderer captures

`rainbow.png` is an actual Vulkanic render output, captured
with the local Release executable using `--capture-hdr`. It is not generated
artwork. The PNG preview applies the renderer's hue-preserving tone map and
linear-to-sRGB conversion to the captured RGB values, at exposure 20. No colour
grading, compositing, sharpening or image upscaling was applied.

The image is 960×540. Its temporary capture config was based on
`config/path_tracer_config.json`, with these showcase overrides:

| Setting | Value |
| --- | --- |
| Camera position | `[0, 2, -10]` |
| Camera look-at | `[-0.35, 2.16, -10.25]` |
| Vertical field of view | 65 degrees |
| Exposure | 20 |
| Primary sky steps (`VIEW_STEPS`) | 4 |
| Secondary ray steps (`Samples`) | 2 |
| Secondary directions (`secondarySamples`) | 2 |
| Scattering orders | 2 |
| Samples per pixel | 1 |
| Warm-up / measured frames | 12 / 12 (24 accumulated frames) |
| Analyzer | Off |
| Rainbow sun direction | `[0.35, 0.01, 0.25]` |

Rain settings remain those in the default config, including both bows, 24 rain
view steps, 4097 angular bins, 500 micrometre effective droplet radius and 0.08
effective variance.

The repository's default config was not edited. Temporary configs, raw HDR
captures and renderer logs are retained locally under the ignored
`cmake-build-ninja/readme-captures` directory. This image illustrates the
renderer; it is not a default-settings screenshot or performance benchmark.
