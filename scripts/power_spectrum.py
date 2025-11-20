#!/usr/bin/env python3
"""
Script to compute power spectrum P(k)
for either the density (from `weight_field.npy`) or the divergence of the 
 - For density: computes the overdensity delta = (count - mean)/mean and
   computes P(k) = <|delta_k|^2>/V, binned isotropically in k.
 - For velocity: computes divergence in Fourier space div_k = i k·v_k and
   P_div(k) = <|div_k|^2>/V.

Outputs: saves k,P(k) arrays and a PNG plot in `outputs/figs/`.

Usage examples:
  python3 scripts/power_spectrum.py --field density --inpath data/preprocessed --boxsize 1000
  python3 scripts/power_spectrum.py --field divv --inpath data/preprocessed --boxsize 1000
  python3 scripts/power_spectrum.py --test
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def radial_bins(kmag, nbins=30, kmin=None, kmax=None):
    """Return bin edges and centers between kmin and kmax."""

    km = np.asarray(kmag)
    if kmax is None:
        kmax = float(km.max())
    pos = km[km > 0]
    if kmin is None or kmin <= 0:
        if pos.size > 0:
            kmin = float(pos.min())
        else:
            kmin = 1e-12
    edges = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean

    return edges, centers


def bin_isotropic(kflat, pflat, edges):
    """Isotropic binning: average pflat per bin in edges. Ignores NaNs and returns (P, N)."""
    k = np.asarray(kflat)
    p = np.asarray(pflat)

    # ignore NaN powers, warn once
    bad = np.isnan(p)
    if bad.any():
        print(f"Warning: found {int(bad.sum())} NaN values in pflat — they will be ignored")
    k = k[~bad]
    p = p[~bad]

    nbins = len(edges) - 1
    inds = np.searchsorted(edges, k, side='right') - 1
    inds = np.clip(inds, 0, nbins - 1)

    sums = np.bincount(inds, weights=p, minlength=nbins)
    counts = np.bincount(inds, minlength=nbins)

    # safe division: avoid runtime warnings from 0/0 by using np.divide
    P = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=P, where=(counts > 0))
    return P, counts.astype(int)


def pk_from_field(field, boxsize, nbins=30, deconv_pcs=False, eps=1e-12):
    """Compute isotropic P(k) for a real 3D field.

    Normalization: P(k) = <|F(k)|^2> / V  (where F is FFT of the field, V=boxsize^3).
    """
    N = field.shape[0]
    dx = boxsize / float(N)  # grid spacing physical units
    vol = boxsize ** 3  # total volume 
    F = np.fft.fftn(field)  # compute the discrete fourier transform
    power = (np.abs(F) ** 2)  # raw squared amplitudes; will scale to P(k) below

    # compute the wavenumber grid in physical units
    k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kx = k1[:, None, None]
    ky = k1[None, :, None]
    kz = k1[None, None, :]
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz) # modulus of k vector

    # optionally deconvolve PCS assignment window by dividing P(k) by |W(k)|^2
    if deconv_pcs:
        # PCS window in 1D approximated as sinc^4 (cubic B-spline)
        # W1D(k) = [sinc(k*dx/2)]^4 where sinc(x)=sin(x)/x.
        # numpy.sinc expects argument in units of pi: sinc(z) = sin(pi*z)/(pi*z)
        argx = (kx * dx / 2.0) / np.pi
        argy = (ky * dx / 2.0) / np.pi
        argz = (kz * dx / 2.0) / np.pi
        W1 = np.sinc(argx) ** 4
        W2 = np.sinc(argy) ** 4
        W3 = np.sinc(argz) ** 4
        Wk = W1 * W2 * W3
        denom = (np.abs(Wk) ** 2) + eps
        power = power / denom

    # Scale raw |F|^2 to physical P(k): multiplying by (dx**3) per FFT gives
    # the continuous FT; thus |F|^2 must be multiplied by dx**6. Divide by
    # the total volume to obtain P(k) = <|delta_k|^2> / V.
    power = power * (dx ** 6) / vol

    kflat = kmag.ravel()
    pflat = power.ravel()
    k_ny = np.pi / dx
    # exclude the k=0 (mean) mode for log binning and build bins up to Nyquist
    mask = kflat > 0
    kflat_nz = kflat[mask]
    pflat_nz = pflat[mask]
    edges, centers = radial_bins(kflat_nz, nbins=nbins, kmax=k_ny)
    P, Nper = bin_isotropic(kflat_nz, pflat_nz, edges)
    return centers, P, Nper





def main():
    p = argparse.ArgumentParser(description='Simple isotropic P(k) calculator')
    p.add_argument('--field', choices=['density'], default='density', help='Only density is supported')
    p.add_argument('--inpath', default='data/preprocessed')
    p.add_argument('--outfig', default='outputs/figs')
    p.add_argument('--boxsize', type=float, default=1000.0)
    p.add_argument('--nbins', type=int, default=30)
    p.add_argument('--deconv-pcs', action='store_true', help='Apply PCS window deconvolution (sinc^4 approximation)')
    p.add_argument('--no-subtract-shot', action='store_true', help='Do NOT subtract shot noise (by default we subtract)')
    args = p.parse_args()

    os.makedirs(args.outfig, exist_ok=True)



    if args.field == 'density':
        #  Run on  pipeline-produced `weight_field.npy` 
        fn_weight = os.path.join(args.inpath, 'weight_field.npy')
        if not os.path.exists(fn_weight):
            raise FileNotFoundError(f'No weight_field found at {fn_weight}')

        arr = np.load(fn_weight).astype(np.float64)
        # detect if arr is already overdensity (mean ~ 0); weight_field is mass-like so we expect non-zero mean
        m = np.nanmean(arr)
        s = np.nanstd(arr)
        is_delta = (abs(m) < 1e-6) or (abs(m) < 1e-3 * s)
        if is_delta:
            delta = np.nan_to_num(arr)
        else:
            if m == 0:
                raise ValueError(f'{fn_weight} has zero mean')
            delta = (arr - m) / m

        # compute isotropic P(k) with optional PCS deconvolution
        k, Pk, Nper = pk_from_field(delta, boxsize=args.boxsize, nbins=args.nbins, deconv_pcs=args.deconv_pcs)

        print("nbins =", len(k))
        print("counts per bin (Nper):", Nper)
        print("nbins non-vuoti:", (Nper>0).sum(), " / ", len(Nper))
        frac_nonempty = float((Nper>0).sum())/len(Nper)
        print(f"Fraction non-empty = {frac_nonempty:.2f}")
        # Shot-noise subtraction: compute N_particles from the sum of PCS weights.
        # For PCS assignment the sum over the whole grid of the weight_field is (approximately) the
        # number of particles (each particle's PCS weights sum to 1 across the grid).
        Np = float(np.nansum(arr))
        if Np <= 0:
            raise ValueError('Computed total particle number Np <= 0, cannot compute shot noise')
        V = args.boxsize ** 3
        P_shot = V / Np

        # Log the particle count and shot noise for reproducibility
        print(f"Total particles (from sum weights) Np = {Np:.0f}")
        print(f"Box volume V = {V:.6e} (Mpc/h)^3, shot noise P_shot = {P_shot:.6e}")

        subtract_shot = not args.no_subtract_shot
        # Preserve the measured P(k) for plotting (before shot subtraction)
        Pk_meas = Pk.copy()
        if subtract_shot:
            Pk_corr = Pk_meas - P_shot
            # avoid negative spurious values after subtraction: set negatives to NaN
            Pk_corr = np.where(Pk_corr > 0, Pk_corr, np.nan)
        else:
            Pk_corr = Pk_meas

        suffix = '_dec' if args.deconv_pcs else ''
        # include nbins in filenames (b{nbins}) and save figures/data in dedicated subfolders
        figs_dir = os.path.join(args.outfig, 'power_spectrum')
        data_dir = os.path.join(args.inpath, 'power_spectrum_data')
        os.makedirs(figs_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        base_name = f'power_weight_field{suffix}_b{args.nbins}'
        base_fig = os.path.join(figs_dir, base_name)
        base_data = os.path.join(data_dir, base_name)
        # save k and the corrected spectrum (after shot-subtraction if enabled) to data/preprocessed
        np.save(base_data + '_k.npy', k)
        np.save(base_data + '_P.npy', Pk_corr)
        print('Saved arrays to', base_data + '_k.npy', 'and', base_data + '_P.npy')

        # --- plotting ---
        dx = args.boxsize / float(arr.shape[0])
        k_ny = np.pi / dx
        k_half_ny = 0.5 * k_ny

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(7, 6))

        # top: measured and corrected P(k)
        ax_top.loglog(k, Pk_meas, '-o', color='orange', label='P_meas (before shot sub)', alpha=0.4)
        ax_top.loglog(k, Pk_corr, '-o', color='green', label='P_corr (after shot sub)')
        ax_top.axvline(k_ny, color='red', linestyle='--', label='k_Nyquist')
        ax_top.axvline(k_half_ny, color='green', linestyle=':', label='0.5 k_Nyquist')
        ax_top.axhline(P_shot, color='k', linestyle='--', label='P_shot')
        ax_top.set_ylabel('P(k)')
        ax_top.set_title('Power: weight_field')

        # annotate Np and P_shot
        txt = f"Np = {Np:.0f}\nP_shot = {P_shot:.3e}"
        ax_top.text(0.98, 0.95, txt, transform=ax_top.transAxes, ha='right', va='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        # bottom: ratio
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = Pk_corr / Pk_meas
        ax_bot.plot(k, ratio, '-o', color = 'purple')
        ax_bot.set_xscale('log')
        ax_bot.set_xlabel('k [h/Mpc]')
        ax_bot.set_ylabel('P_corr/P_meas')
        ax_bot.axhline(1.0, color='0.5', linestyle='--')

        # shade trusted region k < 0.5*k_Nyquist
        # start the shading at the smallest plotted k (first center) instead of an arbitrary tiny value
        k_plot_min = float(k.min()) if np.size(k) > 0 else 1e-12
        ax_top.axvspan(k_plot_min * 0.9, k_half_ny, color='green', alpha=0.08)
        ax_bot.axvspan(k_plot_min * 0.9, k_half_ny, color='green', alpha=0.08)

        ax_top.legend(fontsize=8)
        ax_top.grid(True, which='both', ls=':')
        ax_bot.grid(True, which='both', ls=':')

        # set sensible x-limits so log-axis doesn't extend to extremely small empty region
        x_min = k_plot_min * 0.8
        x_max = float(k.max()) * 1.05
        ax_top.set_xlim(x_min, x_max)
        ax_bot.set_xlim(x_min, x_max)

        outpng = base_fig + '.png'
        plt.tight_layout()
        plt.savefig(outpng, dpi=150)
        plt.close(fig)
        print('Saved plot to', outpng)

        # Save comprehensive .npz with diagnostics and a masked P_corr for trusted bins (to data/preprocessed)
        k_ny = np.pi / dx
        trusted_mask = (k < 0.5 * k_ny) & (Nper >= 10)
        P_corr_masked = np.where(trusted_mask, Pk_corr, np.nan)

        np.savez(base_data + '_all.npz',
             k=k,
             P_meas=Pk_meas,
             P_corr=Pk_corr,
             P_corr_masked=P_corr_masked,
             Nper=Nper,
             k_ny=k_ny,
             trusted_mask=trusted_mask,
             boxsize=args.boxsize,
             Np=Np,
             P_shot=P_shot,
             nbins=args.nbins)
        print('Saved comprehensive arrays to', base_data + '_all.npz')
        return
    
if __name__ == '__main__':
    main()


"""
Se uso --field density nello script power_spectrum.py, su cosa sto calcolando P(k)?
Lo script attuale carica count_field.npy, calcola δ(x) = (count - mean)/mean e poi:
F(k) = FFT[δ(x)]
P(k) = ⟨|F(k)|^2⟩ / V (bin radiale)
Quindi il power è sul campo di sovra-densità δ(x). Se count_field è invece massa assoluta 
per cella, la conversione a δ è necessaria (lo script fa proprio questo).


"""