# Sezione in cui cerco di implementare la classica cosmic structure classification
# come fatto in Hoffman et. al. 2012
 

import argparse
import os
import time
import numpy as np
import sys


def load_fields(path_prefix='data/preprocessed'):
    vx = np.load(os.path.join(path_prefix, 'velocity_field_x.npy'))
    vy = np.load(os.path.join(path_prefix, 'velocity_field_y.npy'))
    vz = np.load(os.path.join(path_prefix, 'velocity_field_z.npy'))
    w = None
    wf = os.path.join(path_prefix, 'weight_field.npy')
    if os.path.exists(wf):
        w = np.load(wf)
    if vx.shape != vy.shape or vx.shape != vz.shape:
        raise ValueError(f'Velocity field shapes do not match: {vx.shape}, {vy.shape}, {vz.shape}')
    else:
        print(f'Loaded velocity fields with shape: {vx.shape}, {vy.shape}, {vz.shape}')
    return vx.astype(np.float64), vy.astype(np.float64), vz.astype(np.float64), w


def kgrid(N, boxsize):
    # physical spacing
    dx = boxsize / float(N)
    k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    # create full 3D grids
    kx = k1[:, None, None]
    ky = k1[None, :, None]
    kz = k1[None, None, :]
    return kx, ky, kz

 
def compute_derivatives_fft(vx, vy, vz, boxsize, sigma_vox=0.0):
    # vx,vy,vz shape (N,N,N)
    N = vx.shape[0]
    assert vx.shape == vy.shape == vz.shape
    dx = boxsize / float(N)
    # Compute FFT
    print('  computing FFTs...')
    t0 = time.time()
    fx = np.fft.fftn(vx)
    fy = np.fft.fftn(vy)
    fz = np.fft.fftn(vz)
    kx, ky, kz = kgrid(N, boxsize)
    k2 = kx * kx + ky * ky + kz * kz  # Squared k modulus

    # Gaussian smoothing in k-space if requested (sigma_vox is in voxels -> convert to physical)
    if sigma_vox and sigma_vox > 0.0:
        sigma_phys = sigma_vox * dx
        print(f'  applying Gaussian smoothing (sigma={sigma_vox} voxels -> {sigma_phys:.6g} Mpc/h)')
        gk = np.exp(-0.5 * (k2 * (sigma_phys ** 2)))
        fx *= gk
        fy *= gk
        fz *= gk

    # derivatives: ifft( i k_j * f )
    ik = 1j
    dvx_dx = np.fft.ifftn(ik * kx * fx).real
    dvx_dy = np.fft.ifftn(ik * ky * fx).real
    dvx_dz = np.fft.ifftn(ik * kz * fx).real

    dvy_dx = np.fft.ifftn(ik * kx * fy).real
    dvy_dy = np.fft.ifftn(ik * ky * fy).real
    dvy_dz = np.fft.ifftn(ik * kz * fy).real

    dvz_dx = np.fft.ifftn(ik * kx * fz).real
    dvz_dy = np.fft.ifftn(ik * ky * fz).real
    dvz_dz = np.fft.ifftn(ik * kz * fz).real

    dt = time.time() - t0
    print(f'  derivatives computed in {dt:.2f} s')
    return dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz


def hoffman_classification(ders, H0, lambda_th=0.0):
    (dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz) = ders
    # S_ij = -1/(2 H0) (d v_i / d x_j + d v_j / d x_i)
    inv2H = -0.5 / H0
    # diagonal elements simplify
    Sxx = - dvx_dx / H0
    Syy = - dvy_dy / H0
    Szz = - dvz_dz / H0
    Sxy = inv2H * (dvx_dy + dvy_dx)
    Sxz = inv2H * (dvx_dz + dvz_dx)
    Syz = inv2H * (dvy_dz + dvz_dy)

    N = Sxx.shape[0]
    Nvox = N * N * N
    print('  assembling shear matrices (flattening)')
    # flatten
    Sxx_f = Sxx.ravel()
    Syy_f = Syy.ravel()
    Szz_f = Szz.ravel()
    Sxy_f = Sxy.ravel()
    Sxz_f = Sxz.ravel()
    Syz_f = Syz.ravel()

    # allocate stacked symmetric matrices for eigval computation
    mats = np.empty((Nvox, 3, 3), dtype=np.float64)
    mats[:, 0, 0] = Sxx_f
    mats[:, 1, 1] = Syy_f
    mats[:, 2, 2] = Szz_f
    mats[:, 0, 1] = Sxy_f
    mats[:, 1, 0] = Sxy_f
    mats[:, 0, 2] = Sxz_f
    mats[:, 2, 0] = Sxz_f
    mats[:, 1, 2] = Syz_f
    mats[:, 2, 1] = Syz_f

    print('  computing eigenvalues')
    t0 = time.time()
    # eigenvalues sorted in ascending order by eigvalsh; we don't need sorting but we will count
    eigs = np.linalg.eigvalsh(mats)
    dt = time.time() - t0
    print(f'  eigenvalues computed in {dt:.2f} s')

    # count eigenvalues > lambda_th (V-web definition per paper)
    counts = np.sum(eigs > lambda_th, axis=1).astype(np.uint8)
    # reshape back
    Nshape = (N, N, N)
    classification = counts.reshape(Nshape)
    eigs0 = eigs[:, 2].reshape(Nshape)  # largest
    eigs1 = eigs[:, 1].reshape(Nshape)
    eigs2 = eigs[:, 0].reshape(Nshape)
    return classification, (eigs0, eigs1, eigs2)


def fill_nans(vx, vy, vz, w=None, method='zero'):
    """
    Fill NaNs in velocity components before FFT. Methods:
      - 'zero': replace NaN with 0
      - 'mean': replace NaN with global mean of each component (computed over non-NaN voxels)
    If a weight field `w` is provided, also reports fraction of empty voxels.
    Returns filled arrays and a dict with statistics.
    """
    stats = {}
    mask = None
    if w is not None:
        mask = (w > 0)
        nvox = w.size
        n_empty = int(np.count_nonzero(~mask))
        stats['n_empty'] = n_empty
        stats['frac_empty'] = float(n_empty) / float(nvox)

    # Work on copies
    vx_f = vx.copy()
    vy_f = vy.copy()
    vz_f = vz.copy()

    # detect NaNs
    nan_vx = np.isnan(vx_f)
    nan_vy = np.isnan(vy_f)
    nan_vz = np.isnan(vz_f)
    any_nan = nan_vx.any() or nan_vy.any() or nan_vz.any()
    stats['had_nan'] = bool(any_nan)

    if not any_nan:
        return vx_f, vy_f, vz_f, stats

    print('  Warning: NaN values detected in velocity fields. Filling using method="%s"' % method)
    if method == 'zero':
        vx_f[nan_vx] = 0.0
        vy_f[nan_vy] = 0.0
        vz_f[nan_vz] = 0.0
    elif method == 'mean':
        # compute nanmean safely
        mvx = np.nanmean(vx_f)
        mvy = np.nanmean(vy_f)
        mvz = np.nanmean(vz_f)
        vx_f[nan_vx] = mvx
        vy_f[nan_vy] = mvy
        vz_f[nan_vz] = mvz
        stats['fill_values'] = (mvx, mvy, mvz)
    else:
        raise ValueError(f'Unknown nan fill method: {method}')

    return vx_f, vy_f, vz_f, stats


def save_outputs(classification, eigs, out_prefix='data/preprocessed', lambda_th=0.0, sigma_vox=0.0, dx=None, extra_meta=None, job_id=None):
    """
    Save classification and eigenvalue maps to files whose names include the chosen
    lambda threshold and Gaussian smoothing. If dx is provided (grid spacing in Mpc/h),
    sigma_vox is converted to physical units for filename clarity.
    """
    import json
    os.makedirs(out_prefix, exist_ok=True)

    if dx is not None:
        sigma_phys = float(sigma_vox) * float(dx)
    else:
        sigma_phys = float(sigma_vox)

    # construct a short suffix using sigma (in voxels) and lambda
    suffix = f"_s{float(sigma_vox):.4g}_l{float(lambda_th):.4g}"

    class_file = os.path.join(out_prefix, f'classification_hoffman{suffix}.npy')
    np.save(class_file, classification)
    print(f'  saved classification -> {class_file}')

    # save eigenvalues with consistent suffix
    for i, arr in enumerate(eigs):
        fname = os.path.join(out_prefix, f'eigenval_{i}_hoffman{suffix}.npy')
        np.save(fname, arr)
        print(f'  saved eigenvalue map -> {fname}')

    # save metadata for reproducibility
    meta = {
        'lambda_th': float(lambda_th),
        'sigma_vox': float(sigma_vox),
        'sigma_phys_Mpc_h': float(sigma_phys),
        'out_prefix': out_prefix,
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_fname = os.path.join(out_prefix, f'metadata_hoffman{suffix}.json')
    with open(meta_fname, 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f'  saved metadata -> {meta_fname}')

def main():
    p = argparse.ArgumentParser(description='V-web Hoffman classifier')
    p.add_argument('--boxsize', type=float, default=1000.0, help='Box size [Mpc/h]')
    p.add_argument('--h', type=float, default=0.6711, help='dimensionless Hubble h (default from header)')
    p.add_argument('--H0', type=float, default=None, help='H0 in km/s/Mpc (overrides h)')
    p.add_argument('--sigma', type=float, default=0.0, help='Gaussian smoothing sigma in voxels')
    p.add_argument('--sigma_phys', type=float, default=None, help='Gaussian smoothing sigma in physical units (Mpc/h). If set, overrides --sigma')
    p.add_argument('--lambda_th', type=float, default=0.0, help='eigenvalue threshold for classification')
    p.add_argument('--nanfill', type=str, default='none', help='How to handle NaN voxels before FFT: "none" (default, abort if NaN), "zero" or "mean"')
    p.add_argument('--inpath', type=str, default='data/preprocessed', help='input folder with velocity_field_*.npy')
    p.add_argument('--outpath', type=str, default='data/preprocessed', help='output folder')
    p.add_argument('--test', action='store_true', help='run a small self-test')
    args = p.parse_args()

    if args.test:
        print('Running quick self-test with small random field (N=32)')   ##############CAMBIARE? 128?
        N = 32
        rng = np.random.RandomState(1)
        vx = rng.normal(scale=100., size=(N, N, N))
        vy = rng.normal(scale=100., size=(N, N, N))
        vz = rng.normal(scale=100., size=(N, N, N))
        H0 = 100.0 * args.h if args.H0 is None else args.H0
        ders = compute_derivatives_fft(vx, vy, vz, args.boxsize, sigma_vox=args.sigma)
        classification, eigs = hoffman_classification(ders, H0, lambda_th=args.lambda_th)
        print('  test classification shape:', classification.shape, 'unique labels:', np.unique(classification))
        return

    print('Loading velocity fields from', args.inpath)
    vx, vy, vz, w = load_fields(args.inpath)
    if w is not None:
        print('weight_field present: min/max/mean', np.nanmin(w), np.nanmax(w), np.nanmean(w))
    N = vx.shape[0]
    # ensure cubic
    if not (vx.shape[0] == vx.shape[1] == vx.shape[2]):
        raise ValueError(f'Velocity fields are not cubic: {vx.shape}')
    print(f'Loaded fields shape: {vx.shape} (N={N})')
    H0 = 100.0 * args.h if args.H0 is None else args.H0
    print(f'Using H0={H0} km/s/Mpc (h={args.h})')

    # grid spacing and smoothing handling
    dx = args.boxsize / float(N)
    print(f'Grid spacing dx = {dx:.6g} Mpc/h (boxsize={args.boxsize}, N={N})')
    sigma_vox = args.sigma
    if args.sigma_phys is not None:
        sigma_vox = float(args.sigma_phys) / dx
        print(f'--sigma_phys provided: {args.sigma_phys} Mpc/h -> sigma={sigma_vox:.6g} voxels')

    # handle NaNs before FFT
    if args.nanfill == 'none':
        any_nan = np.isnan(vx).any() or np.isnan(vy).any() or np.isnan(vz).any()
        if any_nan:
            print('ERROR: NaN values detected in velocity fields. Please implement handling (fill or mask) before running the Hoffman classifier.')
            print('Options: re-run with --nanfill zero or --nanfill mean, or preprocess to remove/mark NaNs.')
            sys.exit(1)
        else:
            vx_f, vy_f, vz_f = vx, vy, vz
            nan_stats = {'had_nan': False}
    else:
        vx_f, vy_f, vz_f, nan_stats = fill_nans(vx, vy, vz, w=w, method=args.nanfill)
        if nan_stats.get('had_nan', False):
            print(f"  NaN fill: method={args.nanfill}, fraction empty={nan_stats.get('frac_empty', 0.0):.6f}")

    tstart = time.time()
    ders = compute_derivatives_fft(vx_f, vy_f, vz_f, args.boxsize, sigma_vox=sigma_vox)
    classification, eigs = hoffman_classification(ders, H0, lambda_th=args.lambda_th)

    # consistency check: trace(S) vs -div(v)/H0
    dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz = ders
    divergence = dvx_dx + dvy_dy + dvz_dz
    trace_from_der = - divergence / H0
    eigs_sum = eigs[0] + eigs[1] + eigs[2]
    diff = eigs_sum - trace_from_der
    print('Consistency check (trace(S) - (-div)/H0):', 'min={:.3e}'.format(np.nanmin(diff)), 'max={:.3e}'.format(np.nanmax(diff)), 'mean={:.3e}'.format(np.nanmean(diff)))

    # Save outputs with lambda and sigma included in filenames
    extra_meta = {
        'boxsize': float(args.boxsize),
        'N': int(N),
        'H0': float(H0),
    }
    save_outputs(classification, eigs, out_prefix=args.outpath, lambda_th=args.lambda_th, sigma_vox=sigma_vox, dx=dx, extra_meta=extra_meta)
    dt = time.time() - tstart
    print(f'Done. Total time: {dt:.2f} s')


if __name__ == '__main__':
    main()



"""
Che fa lo smoothing: applicare un filtro gaussiano elimina il rumore e le piccole 
fluttuazioni locali; in pratica ti vede strutture più grosse (se sigma grande) o più fini 
(se sigma piccolo).
Unità: è fondamentale definire sigma in unità fisiche (Mpc/h) o in voxel. 
Per essere confrontabili tra dataset con risoluzione diversa, è meglio usare sigma in Mpc/h 
e convertire in voxel con sigma_vox = sigma_phys / dx dove dx = boxsize / N.
Effetto su autovalori e classificazione: smoothing riduce l’ampiezza delle derivate, appiattisce 
la distribuzione degli autovalori e tende a diminuire il numero di voxel classificati come “knot” (nodi) 
a favore di filamenti/sheet/void se sigma aumenta. Perciò la λ_th ottimale dipende da sigma.
"""





