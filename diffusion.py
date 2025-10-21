#!/usr/bin/env python3
"""
diffusion.py

Solve 1D diffusion equation using FTCS explicit finite-difference:
    ∂ρ/∂t = D ∂^2ρ/∂x^2

We take D=2. Initial condition: a normalized "box" centered at x=0 (nonzero over a few grid sites).
We record density at several times and verify that the profiles correspond to Gaussians
with sigma(t) = sqrt(2 D t) = 2 * sqrt(t).

Usage:
    python diffusion.py --part=1
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def gaussian(x, sigma):
    """Normalized Gaussian with mean 0 and given sigma."""
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (x / sigma)**2)

def run_diffusion(D=2.0,
                  L=200.0,       # half-domain length (domain -L .. +L)
                  Nx=2001,       # number of grid points (odd so x=0 is included)
                  box_halfwidth=1.0,  # half-width of initial box profile in physical units
                  t_snapshots=None,
                  dt_safety=0.4):
    """
    Run FTCS diffusion and return snapshots and sigma estimates.

    Returns:
      x (array), snapshots (list of rho arrays), times (list), sigma_num (list)
    """
    if t_snapshots is None:
        # pick 5 times where the distribution changes visibly
        t_snapshots = np.array([1.0, 4.0, 9.0, 16.0, 25.0])

    x = np.linspace(-L, L, Nx)
    dx = x[1] - x[0]
    dx2 = dx * dx

    # stability condition for FTCS: D * dt / dx^2 <= 0.5
    dt_max = 0.5 * dx2 / D
    dt = dt_safety * dt_max  # safety factor
    print(f"dx = {dx:.4e}, dt_max = {dt_max:.4e}, using dt = {dt:.4e}")

    # initial profile: normalized box centered at 0 with half-width box_halfwidth
    rho0 = np.zeros_like(x)
    mask = np.abs(x) <= box_halfwidth
    rho0[mask] = 1.0
    # normalize so total mass = 1
    mass = np.sum(rho0) * dx
    rho0 /= mass

    # arrays
    rho = rho0.copy()
    rho_new = np.zeros_like(rho)

    tmax = float(np.max(t_snapshots))
    nsteps = int(np.ceil(tmax / dt))
    print(f"tmax = {tmax}, nsteps = {nsteps}")

    snapshots = []
    times = []
    sigma_num = []

    next_snapshot_idx = 0
    current_time = 0.0

    # Ensure snapshot times sorted
    t_snap_sorted = np.sort(t_snapshots)

    # We'll step until tmax and collect snapshots when crossing snapshot times
    for istep in range(nsteps + 1):
        # record snapshot if we reach or pass the next snapshot time (including t=0 if requested)
        if next_snapshot_idx < len(t_snap_sorted) and current_time >= t_snap_sorted[next_snapshot_idx] - 1e-12:
            # compute numeric sigma via second moment (mean ~ 0)
            sigma2 = np.sum((x**2) * rho) * dx  # variance since mean=0
            sigma_val = np.sqrt(sigma2)
            snapshots.append(rho.copy())
            times.append(current_time)
            sigma_num.append(sigma_val)
            print(f"Snapshot {next_snapshot_idx}: t={current_time:.6f}, sigma_num={sigma_val:.6f}")
            next_snapshot_idx += 1

        # if done all snapshots, can break early if desired
        if current_time >= tmax:
            break

        # FTCS update for interior points
        # rho_new[i] = rho[i] + D*dt/dx2 * (rho[i+1] + rho[i-1] - 2*rho[i])
        rho_new[1:-1] = rho[1:-1] + D * dt / dx2 * (rho[2:] + rho[:-2] - 2.0 * rho[1:-1])

        # boundary conditions: we'll keep rho=0 at boundaries (domain large enough)
        rho_new[0] = 0.0
        rho_new[-1] = 0.0

        # swap
        rho, rho_new = rho_new, rho
        current_time += dt

    # final sanity: ensure we got the requested snapshots; if numerical stepping skipped exact times, we have nearest after times
    return x, snapshots, times, sigma_num

def part1_main(args):
    D = 2.0
    # choose grid and times
    L = args.L
    Nx = args.Nx
    box_halfwidth = args.box_halfwidth
    # default snapshot times if none passed
    if args.snap_times is None:
        t_snapshots = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    else:
        t_snapshots = np.array(args.snap_times, dtype=float)

    x, snapshots, times, sigma_num = run_diffusion(D=D,
                                                  L=L,
                                                  Nx=Nx,
                                                  box_halfwidth=box_halfwidth,
                                                  t_snapshots=t_snapshots)

    # Plot overlays: numerical profile + analytical Gaussian for each snapshot
    plt.figure(figsize=(8, 6))
    for idx, rho in enumerate(snapshots):
        t = times[idx]
        sigma_th = np.sqrt(2 * D * t)  # sigma(t) = sqrt(2 D t)
        sigma_n = sigma_num[idx]
        # analytic Gaussian normalized to unit mass
        gauss = gaussian(x, sigma_th)

        plt.plot(x, rho, label=f"num t={t:.2f}, σ_num={sigma_n:.3f}")
        plt.plot(x, gauss, '--', label=f"analytic σ_th={sigma_th:.3f}")

    plt.xlim(-60, 60)
    plt.xlabel('x')
    plt.ylabel('ρ(x,t)')
    plt.title('Diffusion: numerical profiles and analytic Gaussians')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('diffusion_profiles_overlay.png', dpi=150)
    plt.close()
    print("Saved diffusion_profiles_overlay.png")

    # save each snapshot separately (useful)
    for idx, rho in enumerate(snapshots):
        t = times[idx]
        plt.figure()
        plt.plot(x, rho, label='numerical')
        sigma_th = np.sqrt(2 * D * t)
        plt.plot(x, gaussian(x, sigma_th), '--', label=f'analytic σ={sigma_th:.3f}')
        plt.xlim(-60, 60)
        plt.xlabel('x'); plt.ylabel('ρ')
        plt.title(f'Diffusion snapshot t={t:.3f}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fname = f'diffusion_snapshot_t{t:.3f}.png'.replace('.', 'p')
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")

    # Plot sigma_num vs t and theoretical sigma_th = 2*sqrt(t)
    times = np.array(times)
    sigma_num = np.array(sigma_num)
    sigma_th = np.sqrt(2 * D * times)

    plt.figure()
    plt.plot(times, sigma_num, 'o-', label=r'$\sigma_{\rm num}(t)$')
    plt.plot(times, sigma_th, 's--', label=r'$\sigma_{\rm th}(t)=\sqrt{2Dt}=2\sqrt{t}$')
    plt.xlabel('t')
    plt.ylabel(r'$\sigma$')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('diffusion_sigma_vs_t.png', dpi=150)
    plt.close()
    print("Saved diffusion_sigma_vs_t.png")

    # Optionally perform a simple linear test: sigma_num vs sqrt(t)
    sqrt_t = np.sqrt(times)
    coeffs = np.polyfit(sqrt_t, sigma_num, 1)  # fit sigma_num = a * sqrt(t) + b
    a, b = coeffs
    print(f"Fit sigma_num = a * sqrt(t) + b  -> a={a:.6f}, b={b:.6e}")
    print("Theoretical prefactor should be 2.0 (since sigma = 2 * sqrt(t)).")

    # plot fit
    plt.figure()
    plt.plot(sqrt_t, sigma_num, 'o', label='data')
    plt.plot(sqrt_t, a * sqrt_t + b, '-', label=f'fit: {a:.4f} * sqrt(t) + {b:.4e}')
    plt.plot(sqrt_t, 2.0 * sqrt_t, '--', label='theory: 2.0 * sqrt(t)')
    plt.xlabel(r'$\sqrt{t}$')
    plt.ylabel(r'$\sigma$')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('diffusion_sigma_fit.png', dpi=150)
    plt.close()
    print("Saved diffusion_sigma_fit.png")

def main():
    parser = argparse.ArgumentParser(description="1D diffusion simulation (FTCS).")
    parser.add_argument('--part', type=int, default=1, help='Part number (use 1)')
    parser.add_argument('--L', type=float, default=200.0, help='half-domain length (domain = [-L,L])')
    parser.add_argument('--Nx', type=int, default=2001, help='number of grid points (odd recommended)')
    parser.add_argument('--box_halfwidth', type=float, default=1.0, help='half-width of initial box profile')
    parser.add_argument('--snap_times', nargs='+', type=float,
                        help='times at which to record snapshots (default: 1 4 9 16 25)')
    args = parser.parse_args()

    if args.part == 1:
        part1_main(args)
    else:
        print("Unknown part. Use --part=1")

if __name__ == '__main__':
    main()
