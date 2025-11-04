
"""
Count NaNs in velocity fields produced by preprocessing_data.py.

Usage:
    python3 count_nans.py
"""

import os
import numpy as np

PREP = "data/preprocessed"
vx_f = os.path.join(PREP, "velocity_field_x.npy")
vy_f = os.path.join(PREP, "velocity_field_y.npy")
vz_f = os.path.join(PREP, "velocity_field_z.npy")
w_f  = os.path.join(PREP, "weight_field.npy")

def load_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path)

def main():
    vx = load_safe(vx_f)
    vy = load_safe(vy_f)
    vz = load_safe(vz_f)

    if vx.shape != vy.shape or vx.shape != vz.shape:
        raise ValueError(f"Shape mismatch: {vx.shape}, {vy.shape}, {vz.shape}")

    Nvox = vx.size
    print(f"Loaded velocity fields with shape {vx.shape} -> total voxels = {Nvox:,}")

    # per-component NaNs
    n_nan_vx = int(np.count_nonzero(np.isnan(vx)))
    n_nan_vy = int(np.count_nonzero(np.isnan(vy)))
    n_nan_vz = int(np.count_nonzero(np.isnan(vz)))
    print(f"NaNs per component:")
    print(f"  vx: {n_nan_vx:,} ({n_nan_vx/Nvox:.4%})")
    print(f"  vy: {n_nan_vy:,} ({n_nan_vy/Nvox:.4%})")
    print(f"  vz: {n_nan_vz:,} ({n_nan_vz/Nvox:.4%})")

    # voxels with any NaN in any component
    any_nan_mask = np.isnan(vx) | np.isnan(vy) | np.isnan(vz)
    n_any_nan = int(np.count_nonzero(any_nan_mask))
    print(f"Voxels with any NaN in (vx,vy,vz): {n_any_nan:,} ({n_any_nan/Nvox:.4%})")

    # voxels with all components NaN (completely empty)
    all_nan_mask = np.isnan(vx) & np.isnan(vy) & np.isnan(vz)
    n_all_nan = int(np.count_nonzero(all_nan_mask))
    print(f"Voxels with all components NaN: {n_all_nan:,} ({n_all_nan/Nvox:.4%})")

    # compare with weight_field if exists
    if os.path.exists(w_f):
        w = load_safe(w_f)
        if w.shape != vx.shape:
            print(f"Warning: weight_field shape {w.shape} differs from velocity fields {vx.shape}")
        n_weight_zero = int(np.count_nonzero(w == 0))
        print(f"weight_field: voxels with weight==0: {n_weight_zero:,} ({n_weight_zero/Nvox:.4%})")
        # show agreement between weight==0 and all_nan
        same = int(np.count_nonzero((w == 0) & all_nan_mask))
        print(f"voxels where weight==0 AND all components NaN: {same:,} ({same/Nvox:.4%})")
        # voxels that are all-NaN but weight>0 (unexpected)
        unexpected = int(np.count_nonzero(all_nan_mask & (w != 0)))
        print(f"voxels all-NaN but weight>0 (unexpected): {unexpected:,} ({unexpected/Nvox:.4%})")
    else:
        print("No weight_field.npy found in data/preprocessed (skipping weight checks).")

    # optionally print indices of first few NaN voxels for inspection
    if n_any_nan > 0:
        idxs = np.argwhere(any_nan_mask)
        n_show = min(10, idxs.shape[0])
        print(f"First {n_show} voxel indices with any NaN (zipped as (ix,iy,iz)):")
        for i in range(n_show):
            print(" ", tuple(int(x) for x in idxs[i]))

if __name__ == "__main__":
    main()