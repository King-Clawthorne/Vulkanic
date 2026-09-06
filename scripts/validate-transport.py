"""Run deterministic GPU transport regressions and retain configs, HDRs and logs."""
import argparse
from array import array
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys


def read_hdr(path):
    raw = path.read_bytes()
    width, height = struct.unpack_from('<II', raw)
    pixels = array('f')
    pixels.frombytes(raw[8:])
    if sys.byteorder != 'little':
        pixels.byteswap()
    if len(pixels) != width * height * 4 or not all(map(math.isfinite, pixels)):
        raise RuntimeError(f'Invalid HDR: {path}')
    if any(n < 1 for n in pixels[3::4]):
        raise RuntimeError(f'Unaccumulated pixels: {path}')
    luminance = [0.2126*pixels[i] + 0.7152*pixels[i+1] + 0.0722*pixels[i+2]
                 for i in range(0, len(pixels), 4)]
    return pixels, sum(luminance) / len(luminance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, default=Path('cmake-build-ninja/Vulkanic.exe'))
    parser.add_argument('--output', type=Path, default=Path('cmake-build-ninja/transport-validation'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path('config/path_tracer_config.json').read_text())
    base['render'].update(width=96, height=54, vsync=False)
    base['camera'].update(initialLookAt=[-0.35, 2.16, -10.25], fovYDegrees=65)
    base['rainbow'].update(enabled=1, viewSteps=2, scatteringOrders=1)
    base['sky']['spectralConstants'].update(VIEW_STEPS=2, Samples=1, secondarySamples=2,
                                           SCATTERING_ORDERS=1, SEA_LEVEL_REFRACTIVITY=0,
                                           GROUND_ALBEDO=0)
    results = {}

    def run(name, config):
        prefix = args.output / name
        config_path = prefix.with_suffix('.json').resolve()
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        with prefix.with_suffix('.log').open('w') as log:
            completed = subprocess.run([str(args.exe.resolve()), '--benchmark', '8', '--warmup', '4',
                '--config', str(config_path), '--capture-hdr', str(prefix.with_suffix('.hdrbin').resolve())],
                stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False)
        text = prefix.with_suffix('.log').read_text()
        if completed.returncode or 'VUID-' in text or 'Validation Error' in text:
            raise RuntimeError(f'{name} failed; see {prefix.with_suffix(".log")}')
        pixels, mean = read_hdr(prefix.with_suffix('.hdrbin'))
        results[name] = {'mean_luminance': mean, 'exit_code': completed.returncode,
                         'timings': [line for line in text.splitlines() if 'mean=' in line]}
        print(name, results[name], flush=True)
        return pixels, mean

    previous = None
    for order in range(1, 5):
        c = copy.deepcopy(base)
        c['sky']['spectralConstants']['SCATTERING_ORDERS'] = order
        c['rainbow']['scatteringOrders'] = order
        _, mean = run(f'order-{order}', c)
        if previous is not None and mean <= previous:
            raise RuntimeError('Adding an order must add scattered intensity in this scene')
        previous = mean

    c = copy.deepcopy(base)
    c['sky']['spectralConstants'].update(BETA_R_550=0, BETA_M=0, SCATTERING_ORDERS=1)
    _, one = run('rain-only-1', c)
    c['rainbow']['scatteringOrders'] = 2
    rain_two, two = run('rain-only-2', c)
    if two <= one:
        raise RuntimeError('Rain-to-rain multiple scattering missing')

    c = copy.deepcopy(base)
    c['rainbow'].update(scatteringCoefficient=0, extinctionCoefficient=0)
    enabled, _ = run('zero-rain-enabled', c)
    c['rainbow']['enabled'] = 0
    disabled, _ = run('zero-rain-disabled', c)
    if enabled != disabled:
        raise RuntimeError('Zero-density rain must equal disabled rain')

    # With air scattering absent, changing air order must not change rain.
    c = copy.deepcopy(base)
    c['sky']['spectralConstants'].update(BETA_R_550=0, BETA_M=0, SCATTERING_ORDERS=4)
    c['rainbow']['scatteringOrders'] = 2
    rain_sky4, _ = run('rain-independent-sky-order', c)
    if any(abs(a-b) > 1e-9 + 1e-6*abs(a) for a, b in zip(rain_two, rain_sky4)):
        raise RuntimeError('Atmosphere order changed isolated rain')
    c['rainbow']['viewSteps'] = 7
    rain_steps7, _ = run('rain-own-view-steps', c)
    if rain_steps7 == rain_sky4:
        raise RuntimeError('Rain view steps are not wired to integration')
    c = copy.deepcopy(base)
    c['rainbow']['enabled'] = 0
    sky, _ = run('sky-only-controls-base', c)
    c['rainbow'].update(viewSteps=17, scatteringOrders=4)
    sky_other, _ = run('sky-independent-rain-controls', c)
    if sky != sky_other:
        raise RuntimeError('Disabled rain controls changed sky')
    c = copy.deepcopy(base)
    baseline, _ = run('refraction-base', c)

    c = copy.deepcopy(base)
    c['sky']['spectralConstants']['SEA_LEVEL_REFRACTIVITY'] = 0.000277
    bent, _ = run('refracted', c)
    if bent == baseline:
        raise RuntimeError('Refraction must affect the image')
    c = copy.deepcopy(base)
    c['rainbow']['enabled'] = 0
    c['sky']['spectralConstants'].update(BETA_R_550=0, BETA_M=0,
        SUN_DIRECTION=[1, -0.0069814304, 0], SEA_LEVEL_REFRACTIVITY=0)
    c['camera'].update(initialLookAt=[1, 2.00174533, -10], fovYDegrees=2)
    _, hidden = run('sun-below-geometric-horizon', c)
    c['sky']['spectralConstants']['SEA_LEVEL_REFRACTIVITY'] = 0.000277
    _, visible = run('sun-above-refracted-horizon', c)
    if hidden > 1e-8 or visible <= 1e-4:
        raise RuntimeError('Refraction must reveal the otherwise occluded solar disk')
    (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print('PASS: finite HDRs, validation logs, orders 1-4, rain/rain paths, zero rain, independent controls, refraction.')


if __name__ == '__main__':
    main()
