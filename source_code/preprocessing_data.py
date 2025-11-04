#Script to preprocess data, in particular to extract velocity fields from Quijote snapshots

# Import libraries
import h5py
import hdf5plugin
import numpy as np
import os
from pathlib import Path

#Define parameters
dir_snap = "data/raw_data/snapdir_004"   
BoxSize = 1000.0  # Mpc/h  simulation dimension
grid_size= 128  # grid resolution
file_density_field= "data/raw_data/df_m_128_PCS_z=0.npy"

# Load density field and define the snapshots files
df = np.load(file_density_field)
print(f"Density field loaded: shape={df.shape}, mean={df.mean():.6f}, std={df.std():.6f}")

snapshots_file = sorted([f for f in os.listdir(dir_snap) if f.endswith('.hdf5')])


# Load particles function
def load_particles(snapshot_dir):
    """
    Loads positions and velocities from all snapshot files.
    """
    all_positions = []
    all_velocities = []
    total_particles = 0

    for i, filename in enumerate(snapshots_file):
        filepath= os.path.join(snapshot_dir, filename)

        try: 
            with h5py.File(filepath, 'r') as f:
                if 'PartType1' in f:
                    ptype = f['PartType1']
                    
                    # Positions (convert to Mpc/h)
                    if 'Coordinates' in ptype:
                        pos = ptype['Coordinates'][:]
                        pos = pos / 1000.0  # kpc/h -> Mpc/h
                        all_positions.append(pos)
                        
                    # Velocities (already in km/s)
                    if 'Velocities' in ptype:
                        vel = ptype['Velocities'][:]
                        all_velocities.append(vel)
                        
                    n_particles = len(pos) if 'Coordinates' in ptype else 0
                    total_particles += n_particles
                    print(f"  Particles: {n_particles:,}")

        except Exception as e:
            print(f"  ERRORE nel file {filename}: {e}")
            continue
    
    
    if all_positions and all_velocities:
        positions = np.concatenate(all_positions, axis=0)
        velocities = np.concatenate(all_velocities, axis=0)
        print(f"\nTotal loaded particles: {total_particles:,}")
        print(f"Shape positions: {positions.shape}")
        print(f"Shape velocities: {velocities.shape}")

        # Verify ranges
        print(f"Range positions: X=[{positions[:,0].min():.1f}, {positions[:,0].max():.1f}] Mpc/h")
        print(f"Range velocities: Vx=[{velocities[:,0].min():.1f}, {velocities[:,0].max():.1f}] km/s")

        return positions, velocities
    else:
        raise ValueError("No particles loaded!")




# Create velocity fields function

def pcs_kernel_1d(u):
    u = np.abs(u)
    out = np.zeros_like(u)
    mask1 = u < 1.0
    out[mask1] = (1.0/6.0) * (4.0 - 6.0*u[mask1]**2 + 3.0*u[mask1]**3)
    mask2 = (u >= 1.0) & (u < 2.0)
    out[mask2] = (1.0/6.0) * (2.0 - u[mask2])**3
    return out

def create_velocity_fields(positions, velocities, BoxSize=1000.0, grid_size=128, chunk_size=1000000, periodic=False):
    """
    Creates 3 velocity fields (Vx, Vy, Vz) on a grid.
    """
    print("\nCreate velocity fields")
    print(f"BoxSize: {BoxSize} Mpc/h")
    print(f"Grid size: {grid_size}³")

    # Compute voxel dimension
    voxel_size = BoxSize / grid_size
    G3= grid_size**3
    print(f"Voxel size: {voxel_size:.3f} Mpc/h")

    # Initialize accumulators. 
    sum_vx = np.zeros(G3, dtype=np.float64)
    sum_vy = np.zeros(G3, dtype=np.float64)
    sum_vz = np.zeros(G3, dtype=np.float64)
    sum_w = np.zeros(G3, dtype=np.float64)
    
    # Basic checks
    if positions.shape[0] != velocities.shape[0]:
        raise ValueError(f"Positions and velocities must have same length: {positions.shape[0]} != {velocities.shape[0]}")

        # precompute offsets arrays for the 4 positions along each axis
    offsets = np.array([-1, 0, 1, 2], dtype=np.int32)

    N = positions.shape[0]
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        pos_c = positions[start:end]   # shape (M,3)
        vel_c = velocities[start:end]  # shape (M,3)
        M = pos_c.shape[0]

        # continuous grid positions
        g = pos_c / voxel_size
        i0 = np.floor(g).astype(np.int32)  # shape (M,3)
        frac = g - i0                      # fractional parts (M,3)

        # for each of the 64 neighbor offsets compute flat indices and weights
        for dx in offsets:
            ix = i0[:,0] + dx   # Node indexes along x
            ux = frac[:,0] - dx   # Relative distance from node along x (can be negative)
            wx = pcs_kernel_1d(ux)   # Compute weights along x
            for dy in offsets:
                iy = i0[:,1] + dy
                uy = frac[:,1] - dy
                wy = pcs_kernel_1d(uy)
                for dz in offsets:
                    iz = i0[:,2] + dz
                    uz = frac[:,2] - dz
                    wz = pcs_kernel_1d(uz)

                    # combined 1D weights - filter particles with non zero weight
                    w1 = wx > 0
                    w2 = wy > 0
                    w3 = wz > 0
                    valid_weight = w1 & w2 & w3
                    if not np.any(valid_weight):
                        continue

                    # apply periodic boundaries if requested
                    if periodic:
                        sel = np.nonzero(valid_weight)[0]
                        ixg = (ix[sel] % grid_size).astype(np.int32)
                        iyg = (iy[sel] % grid_size).astype(np.int32)
                        izg = (iz[sel] % grid_size).astype(np.int32)
                        w = (wx[sel] * wy[sel] * wz[sel])
                    else:
                        # only keep indices inside the grid
                        inside = (ix >= 0) & (ix < grid_size) & (iy >= 0) & (iy < grid_size) & (iz >= 0) & (iz < grid_size)
                        good = valid_weight & inside
                        if not np.any(good):
                            continue
                        sel = np.nonzero(good)[0]
                        ixg = ix[sel]
                        iyg = iy[sel]
                        izg = iz[sel]
                        w = (wx[sel] * wy[sel] * wz[sel])

                    flat_idx = (ixg * grid_size + iyg) * grid_size + izg   # ravel index
                    # accumulate using np.bincount 
                    sum_vx_chunk = np.bincount(flat_idx, weights=vel_c[sel,0]*w, minlength=G3)
                    sum_vy_chunk = np.bincount(flat_idx, weights=vel_c[sel,1]*w, minlength=G3)
                    sum_vz_chunk = np.bincount(flat_idx, weights=vel_c[sel,2]*w, minlength=G3)
                    sum_w_chunk  = np.bincount(flat_idx, weights=w, minlength=G3)

                    sum_vx += sum_vx_chunk
                    sum_vy += sum_vy_chunk
                    sum_vz += sum_vz_chunk
                    sum_w  += sum_w_chunk

        # Progress logging per chunk (every 5 chunks)
        chunk_idx = start // chunk_size
        if chunk_idx % 5 == 0:
            print(f"  Processed chunk {chunk_idx}: particles {end}/{N} ({100*end/N:.1f}%)")

    # reshape back
    sum_vx = sum_vx.reshape((grid_size,)*3)
    sum_vy = sum_vy.reshape((grid_size,)*3)
    sum_vz = sum_vz.reshape((grid_size,)*3)
    sum_w  = sum_w.reshape((grid_size,)*3)

    # final velocity grids (float32) — mark empty voxels as NaN
    mask = sum_w > 0
    vx = np.full_like(sum_vx, np.nan, dtype=np.float32)
    vy = np.full_like(sum_vy, np.nan, dtype=np.float32)
    vz = np.full_like(sum_vz, np.nan, dtype=np.float32)
    vx[mask] = (sum_vx[mask] / sum_w[mask]).astype(np.float32)
    vy[mask] = (sum_vy[mask] / sum_w[mask]).astype(np.float32)
    vz[mask] = (sum_vz[mask] / sum_w[mask]).astype(np.float32)
    return vx, vy, vz, sum_w


# Use the functions
positions, velocities = load_particles(dir_snap)
voxel_size = BoxSize / grid_size
velocity_field_x, velocity_field_y, velocity_field_z, weight_field = create_velocity_fields(
    positions, velocities, BoxSize, grid_size, chunk_size=1000000, periodic=True
)
# Save results

output_dir = "data/preprocessed"
os.makedirs(output_dir, exist_ok=True)

files_to_save = [
    (f"{output_dir}/velocity_field_x.npy", velocity_field_x),
    (f"{output_dir}/velocity_field_y.npy", velocity_field_y),
    (f"{output_dir}/velocity_field_z.npy", velocity_field_z),
    (f"{output_dir}/weight_field.npy", weight_field)
]

for filepath, data in files_to_save:
    np.save(filepath, data)
    print(f"Saved: {filepath} - Shape: {data.shape}")

print(f"Velocity fields saved in: {output_dir}")




