#!/usr/bin/env python3
"""
poisson.py

Usage:
  python poisson.py --part=1 [--method=jacobi|sor] [--N=51] [--R=10] [--a=0.6] [--Q=1.0]
  python poisson.py --part=2  # Niter vs tolerance study (Jacobi)
  python poisson.py --part=3  # SOR and scaling with N

Outputs: plots saved to files and a short printout.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import time
from scipy import ndimage

def setup_grid(N=51, R=10.0, a=0.6, Q=1.0):
    """
    Build coordinates and initial fields.
    N: number of grid points per side (odd recommended so origin sits on a grid point)
    R: spherical boundary radius
    a: dipole separation
    Q: magnitude of charges (+Q and -Q)
    Returns: (x,y,z), dx, V, rho, mask_interior
    """
    L = 2.0 * R  # physical box length in each direction
    dx = L / (N - 1)
    coords = np.linspace(-R, R, N)
    x = coords; y = coords; z = coords
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    r = np.sqrt(X**2 + Y**2 + Z**2)

    # Mask for interior points (where we solve). boundary r >= R will be held at 0
    mask_interior = (r < R)

    # charge density rho (point charges approximated by adding charge to nearest grid cell)
    rho = np.zeros((N, N, N), dtype=float)

    # place +Q at z = +a/2 and -Q at z = -a/2, both at x=y=0
    def add_point_charge(charge, pos):
        # pos = (x,y,z) in physical coordinates, find nearest grid index
        ix = int(round((pos[0] + R) / dx))
        iy = int(round((pos[1] + R) / dx))
        iz = int(round((pos[2] + R) / dx))
        # add charge density = Q / (dx^3)
        if 0 <= ix < N and 0 <= iy < N and 0 <= iz < N:
            rho[ix, iy, iz] += charge / (dx**3)
        else:
            print("Charge placed outside grid!")

    add_point_charge(+Q, (0.0, 0.0, +a/2.0))
    add_point_charge(-Q, (0.0, 0.0, -a/2.0))

    # initial potential (start from zero everywhere)
    V = np.zeros_like(rho)

    return (X, Y, Z), dx, V, rho, mask_interior

def jacobi_solver(V, rho, mask_interior, dx, tol=1e-4, maxiter=20000, eps0=1.0, verbose=False):
    """
    Simple Jacobi relaxation for Poisson:
      Laplacian(V) = -rho/eps0
    Update: V_new = (1/6) * sum(neighbor V) + dx^2 * rho/(6*eps0)
    """
    N = V.shape[0]
    V_new = V.copy()
    dx2 = dx*dx
    denom = 6.0
    it = 0
    start = time.time()
    while it < maxiter:
        # shift sums of neighbors
        V_new[1:-1,1:-1,1:-1] = (V[2:,1:-1,1:-1] + V[:-2,1:-1,1:-1] +
                                V[1:-1,2:,1:-1] + V[1:-1,:-2,1:-1] +
                                V[1:-1,1:-1,2:] + V[1:-1,1:-1,:-2]) / denom
        # add source term inside
        V_new[1:-1,1:-1,1:-1] += (dx2 * rho[1:-1,1:-1,1:-1]) / (denom * eps0)

        # enforce spherical boundary V=0 for r>=R by zeroing those points
        V_new[~mask_interior] = 0.0

        # keep the potentials at charge grid points free (we let solver handle them via rho)
        # (optionally could fix singular points, but this is fine numerically)

        maxdiff = np.max(np.abs(V_new - V))
        V, V_new = V_new, V  # swap arrays
        it += 1
        if verbose and it % 200 == 0:
            print(f"Jacobi iter {it} maxdiff {maxdiff:.3e}")
        if maxdiff < tol:
            break
    end = time.time()
    return V, it, maxdiff, end - start

def sor_solver(V, rho, mask_interior, dx, omega=1.8, tol=1e-6, maxiter=20000, eps0=1.0, verbose=False):
    """
    Gauss-Seidel with Successive Over-Relaxation (SOR).
    """
    N = V.shape[0]
    dx2 = dx*dx
    denom = 6.0
    it = 0
    start = time.time()
    # iterate in-place
    while it < maxiter:
        maxdiff = 0.0
        # update only interior points (skip halos and outside sphere)
        for i in range(1, N-1):
            for j in range(1, N-1):
                for k in range(1, N-1):
                    if not mask_interior[i,j,k]:
                        continue
                    neigh_sum = (V[i+1,j,k] + V[i-1,j,k] +
                                 V[i,j+1,k] + V[i,j-1,k] +
                                 V[i,j,k+1] + V[i,j,k-1])
                    V_new = (neigh_sum + dx2 * rho[i,j,k] / eps0) / denom
                    delta = V_new - V[i,j,k]
                    V[i,j,k] += omega * delta
                    if abs(omega * delta) > maxdiff:
                        maxdiff = abs(omega * delta)
        it += 1
        if verbose and it % 50 == 0:
            print(f"SOR iter {it} maxdiff {maxdiff:.3e}")
        if maxdiff < tol:
            break
    end = time.time()
    return V, it, maxdiff, end - start

def radial_average(V, X, Y, Z, bins=50):
    """
    Compute radial average <V>(r) by binning points by their distance from origin.
    Returns r_centers, Vr
    """
    r = np.sqrt(X**2 + Y**2 + Z**2).ravel()
    Vflat = V.ravel()
    rmax = r.max()
    edges = np.linspace(0, rmax, bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    Vr = np.zeros_like(centers)
    counts = np.zeros_like(centers)
    inds = np.digitize(r, edges) - 1
    for idx, val in enumerate(Vflat):
        b = inds[idx]
        if 0 <= b < bins:
            Vr[b] += val
            counts[b] += 1
    nonzero = counts > 0
    Vr[nonzero] /= counts[nonzero]
    return centers, Vr

def analytic_dipole_potential(r, theta, p=1.0, eps0=1.0):
    """
    Dipole potential for point dipole p oriented along z:
      V(r,theta) = (p * cos(theta)) / (4*pi*eps0 * r^2)
    Note: here p = Q*a (choose consistent units)
    """
    import numpy as np
    # avoid division by zero
    r = np.array(r)
    V = np.zeros_like(r, dtype=float)
    mask = r > 1e-12
    V[mask] = (p * np.cos(theta[mask])) / (4.0 * np.pi * eps0 * (r[mask]**2))
    return V

def plot_slice_and_contours(V, X, Y, Z, filename_prefix="poisson"):
    # plot z=0 slice contours (equipotentials)
    N = V.shape[0]
    # find index closest to z=0
    zvals = Z[0,0,:]
    iz = np.argmin(np.abs(zvals - 0.0))
    plt.figure(figsize=(6,5))
    cs = plt.contourf(X[:,:,iz], Y[:,:,iz], V[:,:,iz].T, levels=30)
    plt.colorbar(label='V')
    plt.title('Equipotentials (z=0 slice)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_equipotentials_z0.png", dpi=150)
    plt.close()

def part1(args):
    (X,Y,Z), dx, V, rho, mask_interior = setup_grid(N=args.N, R=args.R, a=args.a, Q=args.Q)
    if args.method == 'jacobi':
        V, it, md, t = jacobi_solver(V, rho, mask_interior, dx, tol=args.tol, maxiter=args.maxiter, verbose=True)
    else:
        V, it, md, t = sor_solver(V, rho, mask_interior, dx, omega=args.omega, tol=args.tol, maxiter=args.maxiter, verbose=True)

    print(f"Converged in {it} iterations, maxdiff {md:.3e}, time {t:.2f}s")

    # equipotentials (z=0)
    plot_slice_and_contours(V, X, Y, Z, filename_prefix=f"poisson_part1_{args.method}")

    # radial average V(r)
    r_centers, Vr = radial_average(V, X, Y, Z, bins=60)

    # analytic large-distance dipole potential along theta=0 (cos=1) for comparison:
    # dipole moment p = Q*a (we used eps0=1 here)
    p = args.Q * args.a
    # construct theta array as average cos(theta) ~ 0 for shells; for simplicity compare magnitude scaling:
    # We'll compare radial decay: Vr * r^2 should approach const ~ p /(4*pi*eps0) if dipole falloff ~ 1/r^2
    plt.figure()
    plt.loglog(r_centers[1:], np.abs(Vr[1:]), 'o', label='numerical <|V|>(r)')
    # overlay 1/r^2 reference
    C = p / (4.0 * np.pi * 1.0)
    r = r_centers[1:]
    plt.loglog(r, C / (r**2), '--', label=r'$\propto 1/r^2$ (dipole)')
    plt.xlabel('r')
    plt.ylabel('V (abs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"poisson_part1_Vr_{args.method}.png", dpi=150)
    plt.close()

    print("Saved equipotential and V(r) plots.")

def part2_iters_vs_tol(args):
    # Fix N, run jacobi for multiple tolerances and record Niter
    (X,Y,Z), dx, V0, rho, mask_interior = setup_grid(N=args.N, R=args.R, a=args.a, Q=args.Q)
    tols = np.logspace(-1, -6, 7)
    niters = []
    for tol in tols:
        V = np.zeros_like(V0)
        Vsol, it, md, t = jacobi_solver(V, rho, mask_interior, dx, tol=tol, maxiter=args.maxiter)
        niters.append(it)
        print(f"tol {tol:.1e} -> iters {it}")
    plt.figure()
    plt.loglog(tols, niters, 'o-')
    plt.gca().invert_xaxis()
    plt.xlabel('tolerance (max change)')
    plt.ylabel('N_iter (Jacobi)')
    plt.title('Iterations vs tolerance (Jacobi)')
    plt.grid(True, which='both', ls=':')
    plt.savefig("poisson_part2_iters_vs_tol.png", dpi=150)
    plt.close()
    print("Saved Niter vs tol plot.")

def part3_sor_and_scaling(args):
    # For fixed accuracy (largest relative change of any point from one iteration to next), study Niter vs N for Jacobi and SOR
    Ns = args.N_list
    tol = args.tol_fixed
    niters_jacobi = []
    niters_sor = []
    times_j = []
    times_s = []
    for N in Ns:
        print(f"Running N={N}")
        (X,Y,Z), dx, V0, rho, mask_interior = setup_grid(N=N, R=args.R, a=args.a, Q=args.Q)

        V = np.zeros_like(V0)
        Vj, itj, mdj, tj = jacobi_solver(V.copy(), rho, mask_interior, dx, tol=tol, maxiter=args.maxiter)
        print(f"Jacobi N={N} iters={itj}")
        niters_jacobi.append(itj)
        times_j.append(tj)

        Vs = np.zeros_like(V0)
        # choose omega heuristically
        omega = args.omega
        Vs, its, mds, ts = sor_solver(Vs, rho, mask_interior, dx, omega=omega, tol=tol, maxiter=args.maxiter)
        print(f"SOR N={N} iters={its}")
        niters_sor.append(its)
        times_s.append(ts)

    plt.figure()
    plt.loglog(Ns, niters_jacobi, 'o-', label='Jacobi')
    plt.loglog(Ns, niters_sor, 's-', label=f'SOR (omega={args.omega})')
    plt.xlabel('N (grid points per side)')
    plt.ylabel('N_iter')
    plt.legend()
    plt.grid(True, which='both', ls=':')
    plt.savefig("poisson_part3_Niter_vs_N.png", dpi=150)
    plt.close()
    print("Saved Niter vs N plot.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=int, default=1)
    parser.add_argument('--method', type=str, default='jacobi', choices=['jacobi', 'sor'])
    parser.add_argument('--N', type=int, default=41, help='grid points per side (odd recommended)')
    parser.add_argument('--R', type=float, default=10.0)
    parser.add_argument('--a', type=float, default=0.6)
    parser.add_argument('--Q', type=float, default=1.0)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--maxiter', type=int, default=20000)
    parser.add_argument('--omega', type=float, default=1.8)
    parser.add_argument('--N_list', nargs='+', type=int, default=[21, 31, 41])
    parser.add_argument('--tol_fixed', type=float, default=1e-4)
    args = parser.parse_args()

    if args.part == 1:
        part1(args)
    elif args.part == 2:
        part2_iters_vs_tol(args)
    elif args.part == 3:
        part3_sor_and_scaling(args)
    else:
        print("Unknown part. Use --part=1,2,3")

if __name__ == '__main__':
    main()
