# Import the libraries 
import os 
import glob
import json
import numpy as np 
import matplotlib.pyplot as plt 

# paths
base = "data/preprocessed"
outfig= "outputs/figs"

h_classification = os.path.join(base, "classification_hoffman.npy")
count = os.path.join(base, "count_field.npy")
w= os.path.join(base, "weight_field.npy")
e0 = os.path.join(base, "eigenval_0.npy")
e1 = os.path.join(base, "eigenval_1.npy")
e2 = os.path.join(base, "eigenval_2.npy")

#Load the data
hoffman_classification = np.load(h_classification)
# avoid shadowing the filename variables: load into clearer names
count_field = np.load(count)
w_field = np.load(w)
e0 = np.load(e0)
e1 = np.load(e1)
e2 = np.load(e2)

# ensure output directory exists
os.makedirs(outfig, exist_ok=True)

# Shape of the data
Nvox = int(hoffman_classification.size)
N = int(hoffman_classification.shape[0])

total_vox = Nvox
print(f"Loaded classification shape: {total_vox} -> Total voxels: {Nvox:,}")

unique_labels, counts_labels = np.unique(hoffman_classification, return_counts=True)
label_names = {0: 'void', 1: 'sheet', 2: 'filament', 3: 'knot'}

print("\nClass distribution:")
for lab, cnt in zip(unique_labels, counts_labels):
    # make sure lab and cnt are plain python ints for formatting
    lab_i = int(lab)
    cnt_i = int(cnt)
    name = label_names.get(lab_i, str(lab_i))
    print(f"  {name} ({lab_i}): {cnt_i:,} voxels ({100.0*cnt_i/Nvox:.2f}%)")
    pct = 100.0 * cnt_i / total_vox
    print(f"   {lab_i} ({name}): {cnt_i:,} voxels ({pct:.3f}%)")

# If weight exists, report zeros
if w_field is not None:
    n_w0 = int(np.count_nonzero(w_field == 0))
    print(f"\nWeight field: voxels with weight==0: {n_w0:,} ({100.0 * n_w0 / total_vox:.4f}%)")


# Statistics of count/density per class 
print("\nStatistics of count_field by class (mean, median, std, 1%, 99%):")
for lab in sorted(np.unique(hoffman_classification)):
    mask = hoffman_classification == lab
    vals = count_field[mask]
    mean = np.mean(vals)
    median = np.median(vals)
    std = np.std(vals)
    p1 = np.percentile(vals, 1)
    p99 = np.percentile(vals, 99)
    print(f"  class {lab}: n={vals.size:,}, mean={mean:.4g}, median={median:.4g}, std={std:.4g}, 1%={p1:.4g}, 99%={p99:.4g}")


# Eigenvalue analysis
if e0 is not None and e1 is not None and e2 is not None:
    e0f = e0.ravel()
    e1f = e1.ravel()
    e2f = e2.ravel()
    sum_e = e0 + e1 + e2
    print("\nEigenvalues summary (global):")
    for arr, name in [(e0f, "e0 (largest)"), (e1f, "e1"), (e2f, "e2"), (sum_e.ravel(), "sum(eigs)")]:
        arr_nonan = arr[~np.isnan(arr)]
        print(f"  {name}: min={arr_nonan.min():.4g}, 1%={np.percentile(arr_nonan,1):.4g}, 50%={np.percentile(arr_nonan,50):.4g}, 99%={np.percentile(arr_nonan,99):.4g}, max={arr_nonan.max():.4g}")

    # plot histograms of eigenvalues
    plt.figure(figsize=(8,6))
    bins = 200
    plt.hist(e0f, bins=bins, alpha=0.6, label='e0', density=True)
    plt.hist(e1f, bins=bins, alpha=0.5, label='e1', density=True)
    plt.hist(e2f, bins=bins, alpha=0.5, label='e2', density=True)
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Eigenvalue distributions (Hoffman)')
    plt.tight_layout()
    plt.savefig(os.path.join(outfig, "eigen_hist_h.png"), dpi=150)
    plt.close()

    # histogram sum(eigs)
    plt.figure(figsize=(6,4))
    plt.hist(sum_e.ravel(), bins=200, color='tab:purple')
    plt.xlabel('sum(eigenvalues)')
    plt.ylabel('Counts')
    plt.title('Distribution of sum(eigenvalues) (Hoffman)')
    plt.tight_layout()
    plt.savefig(os.path.join(outfig, "sum_eigen_hist_h.png"), dpi=150)
    plt.close()


# Visualize classification slices
z= N//2 
print(f"\nVisualizing classification slice at z={z}...")

fig, axs = plt.subplots(1,3, figsize=(18, 5))

# a) Density count
ax = axs[0]
im = ax.imshow(np.log1p(count_field[:, :, z].T), origin='lower', cmap='viridis')
ax.set_title("log(1+count)")
ax.set_xlabel('x (voxels)')
ax.set_ylabel('y (voxels)')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

# b) Classification
ax = axs[1]
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap = ListedColormap(['white', 'lightgray', 'dimgray', 'black'])
norm = BoundaryNorm([ -0.5,0.5,1.5,2.5,3.5 ], cmap.N)
im2 = ax.imshow(hoffman_classification[:,:,z].T, origin='lower', cmap=cmap, norm=norm)
ax.set_title('classification')
ax.set_xlabel('x (voxels)')
ax.set_ylabel('y (voxels)')
cbar = plt.colorbar(im2, ax=ax, ticks=[0,1,2,3], fraction=0.046, pad=0.02)
cbar.ax.set_yticklabels(['void','sheet','filament','knot'])

# c) Sum eigenvalues 
ax = axs[2]
im3 = ax.imshow((e0[:,:,z] + e1[:,:,z] + e2[:,:,z]).T, origin='lower', cmap='RdBu')
ax.set_title('sum(eigs)')
ax.set_xlabel('x (voxels)')
ax.set_ylabel('y (voxels)')
plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.02)

plt.tight_layout()
plt.savefig(os.path.join(outfig, f"slice_comparison_z{z}.png"), dpi=200)
plt.close()
print("Saved slice image:", os.path.join(outfig, f"slice_comparison_z{z}.png"))
