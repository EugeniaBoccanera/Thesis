# FILE df_m_128_PCS_z=0.npy
import numpy as np 
import os
from pathlib import Path
import sys

import h5py
import hdf5plugin

file_path = "/stor/progetti/p1087/p1087u08/Tesi/data/raw_data/df_m_128_PCS_z=0.npy"
snapshot_dir = "data/raw_data/snapdir_004"


def explore_single_file(file_path):
    """Explore a single data file and print basic statistics."""
    print("File Exploration")

    try:
        data = np.load(file_path)
        # Basic informations
        print(f"Shape: {data.shape}")
        print(f"Data Type: {data.dtype}")
        print(f"Total dimensions: {data.size:,} elements")

        # Basic statistics
        print("Basic Statistics:")
        print(f"Mean: {data.mean():.8f}")
        print(f"Std Dev: {data.std():.8f}")
        print(f"Min: {data.min():.8f}")
        print(f"Max: {data.max():.8f}")

        # Sovradensity analysis
        mean_val = data.mean()
        min_val = data.min()

        print(f"Mean field Value: {mean_val:.8f}")
        print(f"Min field Value: {min_val:.8f}")

        is_overdensity = True
        reasons = []
        if abs(mean_val) > 0.01:
            is_overdensity = False
            reasons.append("The mean is significantly different from zero.")
        else: 
            reasons.append("The mean is close to zero.")

        if min_val <-1.0:
            is_overdensity = False
            reasons.append("The minimum value is less than -1.")
        else:
            reasons.append("The minimum value is greater than or equal to -1.")

        print(f"result: {'Overdensity field' if is_overdensity else 'Not an overdensity field'}")
        for reason in reasons:
            print(f" - {reason}")


        # Extra informations 
        print(f"3D shape: {data.shape[0]} x {data.shape[1]} x {data.shape[2]} voxels")
        print(f"memory size: {data.nbytes / (1024**2):.2f} MB")

        print("Sample values : first 5 values per each dimension:")
        print(f"[0:5, 0, 0]:{data[0:5, 0, 0]}")
        print(f"[0, 0:5, 0]:{data[0, 0:5, 0]}")
        print(f"[0, 0, 0:5]:{data[0, 0, 0:5]}")

        return data, is_overdensity

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, False
    


explore_single_file(file_path)



# DIRECTORY snapdir_004 

def explore_single_snapshot(snapshot_file):
    """Explore a single file in a snapshot directory."""
    print(f"Exploring Snapshot Directory: {snapshot_file}")

    try: 
        with h5py.File(snapshot_file, 'r') as f:
            def print_structure(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"GROUP: {name}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"DATASET: {name}")
                    print(f"   Shape: {obj.shape}")
                    print(f"   Type: {obj.dtype}")
                    print(f"   Total elements: {obj.size:,}")
                    if obj.size > 0:
                        print(f"   Memory size: {obj.nbytes / (1024**2):.2f} MB")

            f.visititems(print_structure)

            if 'Header' in f:
                print(f"\nHEADER INFORMATION:")
                header = f['Header']
                for attr_name in header.attrs:
                    attr_value = header.attrs[attr_name]
                    print(f"   {attr_name}: {attr_value}")

            particle_types = [key for key in f.keys() if key.startswith('PartType')]
            if particle_types:
                print(f"\nPARTICLE TYPES:")
                for ptype in particle_types:
                    print(f"   {ptype}:")
                    ptype_group = f[ptype]
                    for dataset_name in ptype_group.keys():
                        dataset = ptype_group[dataset_name]
                        print(f"     - {dataset_name}: {dataset.shape} ({dataset.dtype})")
    except Exception as e:
        print(f"Error reading file: {e}")


def explore_snapshot_directory(snapshot_dir):
    """Explore all files in a snapshot directory."""
    print(f"\nExploring Snapshot Directory")
    print(f"Directory: {snapshot_dir}")

    snapshot_files = sorted([f for f in os.listdir(snapshot_dir) if f.endswith('.hdf5')])

    print(f"Found {len(snapshot_files)} snapshot files:")
    for i, file_name in enumerate(snapshot_files):
        print(f"   {i}: {file_name}")

    first_file = os.path.join(snapshot_dir, snapshot_files[0])
    explore_single_snapshot(first_file)




def test_data_reading(snapshot_dir):
    """Test reading data from snapshot files."""
    print(f"\nTesting Data Reading from Snapshot Directory: {snapshot_dir}")

    snapshot_files = sorted([f for f in os.listdir(snapshot_dir) if f.endswith('.hdf5')])

    for file_name in snapshot_files:
        file_path = os.path.join(snapshot_dir, file_name)
        try:
            with h5py.File(file_path, 'r') as f:
                print(f"\nReading file: {file_name}")
                ptype = f['PartType1']
                
                if 'Coordinates' in ptype:
                    coords = ptype['Coordinates']
                    print(f" Coordinates: {coords.shape}")
                    print(f"   Range X: {coords[:, 0].min():.3f} - {coords[:, 0].max():.3f}")
                    print(f"   Range Y: {coords[:, 1].min():.3f} - {coords[:, 1].max():.3f}")
                    print(f"   Range Z: {coords[:, 2].min():.3f} - {coords[:, 2].max():.3f}")
                
                if 'Velocities' in ptype:
                    velocities = ptype['Velocities']
                    print(f"Velocities : {velocities.shape}")
                    print(f"   Range Vx: {velocities[:, 0].min():.3f} - {velocities[:, 0].max():.3f}")
                    print(f"   Range Vy: {velocities[:, 1].min():.3f} - {velocities[:, 1].max():.3f}")
                    print(f"   Range Vz: {velocities[:, 2].min():.3f} - {velocities[:, 2].max():.3f}")

            


        except Exception as e:
            print(f"Error reading file {file_name}: {e}")




explore_snapshot_directory(snapshot_dir)
test_data_reading(snapshot_dir)