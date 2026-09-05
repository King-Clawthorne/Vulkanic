"""Compare deterministic Vulkanic --capture-hdr files (Python standard library only)."""

import argparse
from array import array
import json
import math
from pathlib import Path
import struct
import sys


def load(path):
    with Path(path).open("rb") as file:
        header = file.read(8)
        if len(header) != 8:
            raise ValueError(f"{path}: missing dimensions")
        width, height = struct.unpack("<II", header)
        values = array("f")
        values.frombytes(file.read())
    if sys.byteorder != "little":
        values.byteswap()
    if not width or not height or len(values) != width * height * 4:
        raise ValueError(f"{path}: invalid HDR dimensions or payload size")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}: non-finite HDR component")
    if any(value < 1 for value in values[3::4]):
        raise ValueError(f"{path}: image contains unaccumulated pixels")
    return (width, height), values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--rtol", type=float, default=2e-4)
    args = parser.parse_args()
    if any(not math.isfinite(t) or t < 0 for t in (args.atol, args.rtol)):
        parser.error("tolerances must be finite and nonnegative")
    dimensions, reference = load(args.reference)
    candidate_dimensions, candidate = load(args.candidate)
    if dimensions != candidate_dimensions:
        raise ValueError("Capture dimensions differ")
    max_error = sum_error = squared_error = squared_reference = 0.0
    failed_components = count_mismatches = 0
    for i, (before, after) in enumerate(zip(reference, candidate)):
        if i % 4 == 3:
            count_mismatches += before != after
            continue
        error = abs(after - before)
        max_error = max(max_error, error)
        sum_error += error
        squared_error += error * error
        squared_reference += before * before
        failed_components += error > args.atol + args.rtol * abs(before)
    passed = failed_components == 0 and count_mismatches == 0
    print(json.dumps({
        "reference": args.reference,
        "candidate": args.candidate,
        "width": dimensions[0], "height": dimensions[1],
        "max_absolute_rgb_error": max_error,
        "mean_absolute_rgb_error": sum_error / (dimensions[0] * dimensions[1] * 3),
        "relative_l2_rgb_error": math.sqrt(squared_error / max(squared_reference, 1e-30)),
        "failed_components": failed_components,
        "accumulation_count_mismatches": count_mismatches,
        "absolute_tolerance": args.atol, "relative_tolerance": args.rtol,
        "passed": passed,
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"HDR comparison failed: {error}", file=sys.stderr)
        sys.exit(1)
