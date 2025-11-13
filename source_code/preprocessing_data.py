#Script to preprocess data, in particular to extract velocity fields from Quijote snapshots

# Import libraries
import h5py
import hdf5plugin
import numpy as np
import os
from pathlib import Path

# Define parameters
dir_snap = "data/raw_data/snapdir_004"   
BoxSize = 1000.0  # Mpc/h  simulation dimension
grid_size= 128  # grid resolution
file_density_field= "data/raw_data/df_m_128_PCS_z=0.npy"

# Load density field and define the snapshots files
df = np.load(file_density_field)
print(f"Density field loaded: shape={df.shape}, mean={df.mean():.6f}, std={df.std():.6f}")

snapshots_file = sorted([f for f in os.listdir(dir_snap) if f.endswith('.hdf5')])


def load_particles(snapshot_dir):
    """Read all PartType1 positions and velocities from snapshot files.

    Returns:
      positions: (N,3) in Mpc/h
      velocities: (N,3) in km/s
    """
    all_positions = []
    all_velocities = []
    total = 0
    for filename in snapshots_file:
        filepath = os.path.join(snapshot_dir, filename)
        try:
            with h5py.File(filepath, 'r') as f:
                if 'PartType1' in f:
                    p = f['PartType1']
                    if 'Coordinates' in p:
                        pos = p['Coordinates'][:] / 1000.0  # kpc/h -> Mpc/h
                        all_positions.append(pos)
                    if 'Velocities' in p:
                        vel = p['Velocities'][:]
                        all_velocities.append(vel)
                    n = pos.shape[0] if 'Coordinates' in p else 0
                    total += n
                    print(f"  Loaded {n:,} particles from {filename}")
        except Exception as e:
            print(f"  ERRORE nel file {filename}: {e}")
            continue
    if not all_positions or not all_velocities:
        raise ValueError('No particles loaded from snapshots')
    positions = np.concatenate(all_positions, axis=0)
    velocities = np.concatenate(all_velocities, axis=0)
    print(f"Total loaded particles: {total:,}")
    return positions, velocities


def pcs_kernel_1d(u):
    """Piecewise-cubic spline (PCS) 1D kernel.
    Accepts array-like u (can be negative); returns kernel value(s).
    """
    u = np.abs(u)
    out = np.zeros_like(u)
    mask1 = u < 1.0
    out[mask1] = (1.0/6.0) * (4.0 - 6.0*u[mask1]**2 + 3.0*u[mask1]**3)
    mask2 = (u >= 1.0) & (u < 2.0)
    out[mask2] = (1.0/6.0) * (2.0 - u[mask2])**3
    return out


def pcs_weights_for_fracs(frac, offsets=np.array([-1, 0, 1, 2], dtype=np.int32)):
    """Vectorized PCS weights for a chunk.

    frac: (M,3) fractional positions inside node
    returns: wx, wy, wz arrays of shape (M, len(offsets))
    where wx[i,j] = pcs_kernel_1d(frac[i,0] - offsets[j])
    """
    offs = offsets[None, :]
    ux = frac[:, 0][:, None] - offs
    uy = frac[:, 1][:, None] - offs
    uz = frac[:, 2][:, None] - offs
    wx = pcs_kernel_1d(ux)
    wy = pcs_kernel_1d(uy)
    wz = pcs_kernel_1d(uz)
    return wx, wy, wz


def _accumulate_chunk(i0, vel_c, wx, wy, wz, offsets, grid_size, periodic):
    """Accumulate bincounts for a chunk.

    Returns tuple of (sum_vx_chunk, sum_vy_chunk, sum_vz_chunk, sum_w_chunk, sum_counts_chunk)
    each shaped (G3,)
    """
    G3 = grid_size**3
    sum_vx_chunk = np.zeros(G3, dtype=np.float64)
    sum_vy_chunk = np.zeros(G3, dtype=np.float64)
    sum_vz_chunk = np.zeros(G3, dtype=np.float64)
    sum_w_chunk = np.zeros(G3, dtype=np.float64)
    sum_counts_chunk = np.zeros(G3, dtype=np.float64)

    M = i0.shape[0]
    for ix_idx, dx in enumerate(offsets):
        ix = i0[:, 0] + dx
        wx_col = wx[:, ix_idx]
        for iy_idx, dy in enumerate(offsets):
            iy = i0[:, 1] + dy
            wy_col = wy[:, iy_idx]
            for iz_idx, dz in enumerate(offsets):
                iz = i0[:, 2] + dz
                wz_col = wz[:, iz_idx]

                valid = (wx_col > 0) & (wy_col > 0) & (wz_col > 0)
                if not np.any(valid):
                    continue
                sel = np.nonzero(valid)[0]

                if periodic:
                    ixg = (ix[sel] % grid_size).astype(np.int64)
                    iyg = (iy[sel] % grid_size).astype(np.int64)
                    izg = (iz[sel] % grid_size).astype(np.int64)
                else:
                    inside = (ix >= 0) & (ix < grid_size) & (iy >= 0) & (iy < grid_size) & (iz >= 0) & (iz < grid_size)
                    good = valid & inside
                    if not np.any(good):
                        continue
                    sel = np.nonzero(good)[0]
                    ixg = ix[sel]
                    iyg = iy[sel]
                    izg = iz[sel]

                flat_idx = (ixg * grid_size + iyg) * grid_size + izg
                w = (wx_col[sel] * wy_col[sel] * wz_col[sel])

                sum_vx_chunk += np.bincount(flat_idx, weights=vel_c[sel, 0] * w, minlength=G3)
                sum_vy_chunk += np.bincount(flat_idx, weights=vel_c[sel, 1] * w, minlength=G3)
                sum_vz_chunk += np.bincount(flat_idx, weights=vel_c[sel, 2] * w, minlength=G3)
                sum_w_chunk += np.bincount(flat_idx, weights=w, minlength=G3)
                sum_counts_chunk += np.bincount(flat_idx, weights=np.ones_like(flat_idx, dtype=np.float64), minlength=G3)

    return sum_vx_chunk, sum_vy_chunk, sum_vz_chunk, sum_w_chunk, sum_counts_chunk


# Load particles function
## load_particles is defined above (renamed from read_snapshots). The function
## body includes loading from all snapshot files and returning positions and
## velocities.




# Create velocity fields function

# Note: pcs_kernel_1d is defined earlier; reuse that implementation

def create_velocity_fields(positions, velocities, BoxSize=1000.0, grid_size=128, chunk_size=1000000, periodic=False):
    """
    Create 3 velocity fields (Vx, Vy, Vz) on a regular grid using PCS assignment.

    Parameters
    ----------
    positions : ndarray, shape (N,3)
        Particle positions in the same length units used for BoxSize. By
        convention in this script positions are expected in Mpc/h (the
        loader converts kpc/h -> Mpc/h).
    velocities : ndarray, shape (N,3)
        Particle velocities in km/s.
    BoxSize : float
        Box size in Mpc/h.
    grid_size : int
        Number of grid cells per side.
    chunk_size : int
        Number of particles to process per chunk to limit memory usage.
    periodic : bool
        Whether to apply periodic boundary conditions when assigning to the grid.

    Returns
    -------
    vx, vy, vz : ndarray, shape (grid_size,grid_size,grid_size)
        Mass-weighted velocity components on the grid (float32). Empty voxels
        are set to NaN.
    weight_field : ndarray, shape (grid_size,grid_size,grid_size)
        Sum of PCS weights assigned to each voxel (float64). For equal-mass
        particles this is proportional to mass per voxel; multiplying by a
        particle mass (from snapshot header MassTable) produces a mass field.
    count_field : ndarray, shape (grid_size,grid_size,grid_size)
        Integer particle counts per voxel (produced by rounding the float
        accumulator). Note that due to the PCS kernel a single particle can
        contribute to multiple voxels, so sum(count_field) should be close to
        the number of particles but may differ at domain boundaries without
        periodic=True.
    """
    print("\nCreate velocity fields")
    print(f"BoxSize: {BoxSize} Mpc/h")
    print(f"Grid size: {grid_size}³")

    # Compute voxel dimension
    voxel_size = BoxSize / grid_size
    G3= grid_size**3
    print(f"Voxel size: {voxel_size:.3f} Mpc/h")

    # Initialize accumulators 
    sum_vx = np.zeros(G3, dtype=np.float64)
    sum_vy = np.zeros(G3, dtype=np.float64)
    sum_vz = np.zeros(G3, dtype=np.float64)
    sum_w = np.zeros(G3, dtype=np.float64)
    # additional accumulator to produce count field
    # use float accumulator because np.bincount with 'weights' returns float64
    sum_counts = np.zeros(G3, dtype=np.float64)
    
    # Basic checks
    if positions.shape[0] != velocities.shape[0]:
        raise ValueError(f"Positions and velocities must have same length: {positions.shape[0]} != {velocities.shape[0]}")

    # precompute offsets arrays for the 4 positions along each axis
    offsets = np.array([-1, 0, 1, 2], dtype=np.int32)

    N = positions.shape[0]
    # Process particles in chunks
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        pos_c = positions[start:end]   # shape (M,3)
        vel_c = velocities[start:end]  # shape (M,3)
        M = pos_c.shape[0]

        # continuous grid positions
        g = pos_c / voxel_size
        i0 = np.floor(g).astype(np.int32)  # shape (M,3) - integer part
        frac = g - i0                      # fractional parts (M,3)

        # Vectorized PCS weights for the chunk (M,4) arrays
        wx, wy, wz = pcs_weights_for_fracs(frac, offsets=offsets)

        # Accumulate the chunk using the helper (returns flattened G3 arrays)
        sum_vx_chunk, sum_vy_chunk, sum_vz_chunk, sum_w_chunk, sum_counts_chunk = _accumulate_chunk(
            i0, vel_c, wx, wy, wz, offsets, grid_size, periodic
        )

        sum_vx += sum_vx_chunk
        sum_vy += sum_vy_chunk
        sum_vz += sum_vz_chunk
        sum_w  += sum_w_chunk
        sum_counts += sum_counts_chunk

        # Progress logging per chunk (every 5 chunks)
        chunk_idx = start // chunk_size
        if chunk_idx % 5 == 0:
            print(f"  Processed chunk {chunk_idx}: particles {end}/{N} ({100*end/N:.1f}%)")

    # reshape back
    sum_vx = sum_vx.reshape((grid_size,)*3)
    sum_vy = sum_vy.reshape((grid_size,)*3)
    sum_vz = sum_vz.reshape((grid_size,)*3)
    sum_w  = sum_w.reshape((grid_size,)*3)
    sum_counts = sum_counts.reshape((grid_size,)*3)
    # round and cast to integer counts (each particle contributed integer counts)
    sum_counts = np.rint(sum_counts).astype(np.int64)

    # final velocity grids (float32) — mark empty voxels as NaN
    mask = sum_w > 0
    vx = np.full_like(sum_vx, np.nan, dtype=np.float32)
    vy = np.full_like(sum_vy, np.nan, dtype=np.float32)
    vz = np.full_like(sum_vz, np.nan, dtype=np.float32)
    vx[mask] = (sum_vx[mask] / sum_w[mask]).astype(np.float32)
    vy[mask] = (sum_vy[mask] / sum_w[mask]).astype(np.float32)
    vz[mask] = (sum_vz[mask] / sum_w[mask]).astype(np.float32)
    # return vx, vy, vz, sum_w, count_field
    return vx, vy, vz, sum_w, sum_counts


# Use the functions
positions, velocities = load_particles(dir_snap)
voxel_size = BoxSize / grid_size
velocity_field_x, velocity_field_y, velocity_field_z, weight_field, count_field = create_velocity_fields(
    positions, velocities, BoxSize, grid_size, chunk_size=1000000, periodic=True
)

# Save results
output_dir = "data/preprocessed"
os.makedirs(output_dir, exist_ok=True)

files_to_save = [
    (f"{output_dir}/velocity_field_x.npy", velocity_field_x),
    (f"{output_dir}/velocity_field_y.npy", velocity_field_y),
    (f"{output_dir}/velocity_field_z.npy", velocity_field_z),
    (f"{output_dir}/weight_field.npy", weight_field),
    (f"{output_dir}/count_field.npy", count_field)
]

for filepath, data in files_to_save:
    np.save(filepath, data)
    print(f"Saved: {filepath} - Shape: {data.shape}")

# mass_field is not created (masses are not used here); use MassTable from snapshot header

print(f"Velocity fields saved in: {output_dir}")


