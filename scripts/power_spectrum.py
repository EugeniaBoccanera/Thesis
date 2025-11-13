#!/usr/bin/env python3
"""
Script to compute power spectrum P(k)
for either the density (from `weight_field.npy`) or the divergence of the
velocity field (div v from `velocity_field_{x,y,z}.npy`).

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


def radial_bins(kmag, nbins, kmin, kmax):
    """Return bin edges and centers between kmin and kmax."""
    edges = np.linspace(kmin, kmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def bin_isotropic(kflat, pflat, edges):
    """Simple isotropic binning: compute average P in each radial bin."""
    inds = np.digitize(kflat, edges) - 1
    nbins = len(edges) - 1
    P = np.zeros(nbins)
    N = np.zeros(nbins, dtype=int)
    for i in range(nbins):
        sel = inds == i
        N[i] = sel.sum()
        if N[i] > 0:
            P[i] = pflat[sel].mean()
        else:
            P[i] = np.nan
    return P, N


def pk_from_field(field, boxsize, nbins=30, deconv_pcs=False, eps=1e-12):
    """Compute isotropic P(k) for a real 3D field.

    Normalization: P(k) = <|F(k)|^2> / V  (where F is FFT of the field, V=boxsize^3).
    """
    N = field.shape[0]
    dx = boxsize / float(N)
    vol = boxsize ** 3
    F = np.fft.fftn(field)
    power = (np.abs(F) ** 2) / vol

    k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kx = k1[:, None, None]
    ky = k1[None, :, None]
    kz = k1[None, None, :]
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)

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

    kflat = kmag.ravel()
    pflat = power.ravel()
    k_ny = np.pi / dx
    edges, centers = radial_bins(kflat, nbins, 0.0, k_ny)
    P, Nper = bin_isotropic(kflat, pflat, edges)
    return centers, P, Nper


def pk_divergence(vx, vy, vz, boxsize, nbins=30):
    """Compute P(k) for the divergence field (in Fourier space).

    div_k = i * (k·v_k), so we compute |div_k|^2 / V and bin isotropically.
    """
    N = vx.shape[0]
    dx = boxsize / float(N)
    vol = boxsize ** 3
    Fx = np.fft.fftn(vx)
    Fy = np.fft.fftn(vy)
    Fz = np.fft.fftn(vz)
    k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kx = k1[:, None, None]
    ky = k1[None, :, None]
    kz = k1[None, None, :]
    divk = 1j * (kx * Fx + ky * Fy + kz * Fz)
    power = (np.abs(divk) ** 2) / vol
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    kflat = kmag.ravel()
    pflat = power.ravel()
    k_ny = np.pi / dx
    edges, centers = radial_bins(kflat, nbins, 0.0, k_ny)
    P, Nper = bin_isotropic(kflat, pflat, edges)
    return centers, P, Nper


def main():
    p = argparse.ArgumentParser(description='Simple isotropic P(k) calculator')
    p.add_argument('--field', choices=['density', 'divv'], default='density')
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
        base = os.path.join(args.outfig, f'power_weight_field{suffix}')
        # save k and the corrected spectrum (after shot-subtraction if enabled)
        np.save(base + '_k.npy', k)
        np.save(base + '_P.npy', Pk_corr)
        print('Saved arrays to', base + '_k.npy and _P.npy')

        # --- plotting ---
        dx = args.boxsize / float(arr.shape[0])
        k_ny = np.pi / dx
        k_half_ny = 0.5 * k_ny

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(7, 6))

        # top: measured and corrected P(k)
        ax_top.loglog(k, Pk_meas, '-o', label='P_meas (before shot sub)', alpha=0.4)
        ax_top.loglog(k, Pk_corr, '-o', label='P (after shot sub)')
        ax_top.axvline(k_ny, color='0.5', linestyle='--', label='k_Nyquist')
        ax_top.axvline(k_half_ny, color='0.7', linestyle=':', label='0.5 k_Nyquist')
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
        ax_bot.plot(k, ratio, '-o')
        ax_bot.set_xscale('log')
        ax_bot.set_xlabel('k [h/Mpc]')
        ax_bot.set_ylabel('P_corr/P_meas')
        ax_bot.axhline(1.0, color='0.5', linestyle='--')

        # shade trusted region k < 0.5*k_Nyquist
        ax_top.axvspan(1e-12, k_half_ny, color='green', alpha=0.08)
        ax_bot.axvspan(1e-12, k_half_ny, color='green', alpha=0.08)

        ax_top.legend(fontsize=8)
        ax_top.grid(True, which='both', ls=':')
        ax_bot.grid(True, which='both', ls=':')

        outpng = base + '.png'
        plt.tight_layout()
        plt.savefig(outpng, dpi=150)
        plt.close(fig)
        print('Saved plot to', outpng)
        return
    else:
        vx = np.load(os.path.join(args.inpath, 'velocity_field_x.npy'))
        vy = np.load(os.path.join(args.inpath, 'velocity_field_y.npy'))
        vz = np.load(os.path.join(args.inpath, 'velocity_field_z.npy'))
        k, Pk, Nper = pk_divergence(vx, vy, vz, boxsize=args.boxsize, nbins=args.nbins)
        base = os.path.join(args.outfig, 'power_divv')
        np.save(base + '_k.npy', k)
        np.save(base + '_P.npy', Pk)
        print('Saved arrays to', base + '_k.npy and _P.npy')
        plt.figure()
        plt.loglog(k, Pk, '-o')
        plt.xlabel('k [h/Mpc]')
        plt.ylabel('P(k)')
        plt.title('Power: divv')
        dx = args.boxsize / float(vx.shape[0])
        k_ny = np.pi / dx
        plt.axvline(k_ny, color='0.5', linestyle='--', label='k_Nyquist')
        plt.legend()
        outpng = base + '.png'
        plt.grid(True, which='both', ls=':')
        plt.tight_layout()
        plt.savefig(outpng, dpi=150)
        plt.close()
        print('Saved plot to', outpng)


if __name__ == '__main__':
    main()


"""
Se uso --field density nello script power_spectrum.py, su cosa sto calcolando P(k)?
Lo script attuale carica count_field.npy, calcola δ(x) = (count - mean)/mean e poi:
F(k) = FFT[δ(x)]
P(k) = ⟨|F(k)|^2⟩ / V (bin radiale)
Quindi il power è sul campo di sovra-densità δ(x). Se count_field è invece massa assoluta 
per cella, la conversione a δ è necessaria (lo script fa proprio questo).

Se uso --field divv, su cosa calcolo P(k)?
divv calcola il power spectrum della divergenza del campo di velocità:
v_x,v_y,v_z → FFT → div_k = i (k·v_k)
P_div(k) = ⟨|div_k|^2⟩ / V
Questo è il power spectrum del campo divergente (scala della convergenza/espansione del flusso),
 non il power spectrum diretto della densità. È utile perché la derivata accentua le componenti 
 ad alta k e ti permette di vedere problemi nelle derivate (es. rumore amplificato).
"""