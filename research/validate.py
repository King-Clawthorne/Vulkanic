import pathlib
import numpy as np
import pandas as pd


def main() -> None:
    # Resolve the result files relative to this script so it works from any cwd.
    out = pathlib.Path(__file__).resolve().parent / "results/gpu_validation.csv"
    
    # The component columns are ordered as RGB, then I/Q/U/V for each color.
    cols = [f"{color}_{component}" for color in ("red", "green", "blue") for component in "IQUV"]
    keys = ["sun_elevation_deg", "view_elevation_deg"]
    cpu_cols = ["cpu_" + c for c in cols]
    
    # Keep the latest CPU reference for each geometry, then attach it to each GPU row.
    cpu = pd.read_csv(out.with_name("cpu_reference_stokes.csv")).drop_duplicates(keys, keep="last")
    d = pd.read_csv(out.with_name("gpu_sweep.csv")).merge(cpu[keys + cpu_cols], on=keys, how="left")
    
    # Reshape flat CSV columns into (row, color, Stokes component) arrays.
    v, r = (d[c].to_numpy(dtype=float).reshape(-1, 3, 4) for c in (cols, cpu_cols))
    diff, r_norm = v - r, np.linalg.norm(r, axis=(1, 2))
    
    # Normalize by the corresponding CPU signal, with a tiny floor for zero references.
    d["relative_intensity_error_vs_cpu"] = np.linalg.norm(diff[:, :, 0], axis=1) / np.maximum(np.linalg.norm(r[:, :, 0], axis=1), 1e-300)
    d["relative_stokes_error_vs_cpu"] = np.linalg.norm(diff, axis=(1, 2)) / np.maximum(r_norm, 1e-300)
    d["cpu_reference_stokes_norm"] = r_norm
    
    # Preserve source columns, append the three derived metrics, and retain CPU values.
    d[[*d.columns[:-15], *d.columns[-3:], *cpu_cols]].to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
