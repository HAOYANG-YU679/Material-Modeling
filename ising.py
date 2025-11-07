#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time

def init_spins(n, mode='random'):
    if mode == 'random':
        return np.where(np.random.rand(n, n) < 0.5, 1, -1).astype(np.int8)
    elif mode == 'up':
        return np.ones((n, n), dtype=np.int8)
    else:
        return np.ones((n, n), dtype=np.int8)

def total_energy(spins, J):
    E = -J * np.sum(spins * np.roll(spins, -1, axis=1))
    E += -J * np.sum(spins * np.roll(spins, -1, axis=0))
    return E

def delta_energy_flip(spins, i, j, J):
    n = spins.shape[0]
    s = spins[i, j]
    neigh = spins[(i+1) % n, j] + spins[(i-1) % n, j] + spins[i, (j+1) % n] + spins[i, (j-1) % n]
    return 2.0 * J * s * neigh

def metropolis_sweep(spins, beta, J, boltzmann_lookup=None):
    n = spins.shape[0]
    N = n * n
    for _ in range(N):
        i = np.random.randint(n)
        j = np.random.randint(n)
        dE = delta_energy_flip(spins, i, j, J)
        if dE <= 0:
            spins[i, j] = -spins[i, j]
        else:
            if boltzmann_lookup is not None:
                p = boltzmann_lookup.get(round(dE,8), np.exp(-beta * dE))
            else:
                p = np.exp(-beta * dE)
            if np.random.rand() < p:
                spins[i, j] = -spins[i, j]

def run_temperature(n, T, J=1.5, equil_sweeps=500, meas_sweeps=1000, init_mode='random'):
    kB = 1.0
    beta = 1.0 / (kB * T)
    spins = init_spins(n, mode=init_mode)
    boltzmann_lookup = {0.0: 1.0, 4.0*J: np.exp(-beta*4.0*J), 8.0*J: np.exp(-beta*8.0*J)}
    for _ in range(equil_sweeps):
        metropolis_sweep(spins, beta, J, boltzmann_lookup)
    E_acc = 0.0
    E2_acc = 0.0
    M_acc = 0.0
    Mabs_acc = 0.0
    samples = 0
    for _ in range(meas_sweeps):
        metropolis_sweep(spins, beta, J, boltzmann_lookup)
        E = total_energy(spins, J)
        M = np.sum(spins)
        E_acc += E
        E2_acc += E * E
        M_acc += M
        Mabs_acc += abs(M)
        samples += 1
    E_mean = E_acc / samples
    E2_mean = E2_acc / samples
    M_mean = M_acc / samples
    Mabs_mean = Mabs_acc / samples
    N = n * n
    C_per_spin = (E2_mean - E_mean * E_mean) / (kB * T * T * N)
    return {'E_mean': E_mean, 'E2_mean': E2_mean,
            'M_mean': M_mean / N, 'Mabs_mean': Mabs_mean / N,
            'C_per_spin': C_per_spin}

def part1_magnetization_vs_T(n=50, J=1.5):
    Tc_est = 2.269185314 * J
    print(f"(estimate) Tc ≈ {Tc_est:.3f} for J={J}")
    T_vals = np.concatenate([
        np.linspace(1.0, 2.8, 10),
        np.linspace(2.8, 4.0, 25),
        np.linspace(4.0, 6.0, 8)
    ])
    T_vals = np.unique(np.round(T_vals, 3))
    equil_sweeps = 800
    meas_sweeps = 1600
    stats = []
    t0 = time()
    for T in T_vals:
        s = run_temperature(n, T, J=J, equil_sweeps=equil_sweeps, meas_sweeps=meas_sweeps)
        s['T'] = T
        stats.append(s)
        print(f"T={T:.3f}   |M|/N={s['Mabs_mean']:.4f}   C/N={s['C_per_spin']:.4f}")
    df = pd.DataFrame(stats).sort_values('T')
    # plot
    plt.figure(figsize=(7,5))
    plt.plot(df['T'], df['Mabs_mean'], marker='o')
    plt.xlabel('T (dimensionless)')
    plt.ylabel('|M|/N')
    plt.title(f'n={n} magnetization vs T (J={J})')
    plt.grid(True)
    plt.show()
    return df

def part2_specific_heat_sizes(sizes=[5,10,20,30,40,50], J=1.5):
    Tscan = np.linspace(2.2, 4.6, 21)
    equil_sweeps = 400
    meas_sweeps = 1000
    C_vs_T = {}
    for n in sizes:
        rows = []
        print(f"Running size n={n} ...")
        for T in Tscan:
            s = run_temperature(n, T, J=J, equil_sweeps=equil_sweeps, meas_sweeps=meas_sweeps)
            rows.append({'T': T, 'C_per_spin': s['C_per_spin']})
        C_vs_T[n] = pd.DataFrame(rows)
    rows = []
    for n in sizes:
        df = C_vs_T[n]
        idx = df['C_per_spin'].idxmax()
        rows.append({'n': n, 'Cmax_per_spin': df.loc[idx,'C_per_spin'], 'T_at_max': df.loc[idx,'T']})
    dfC = pd.DataFrame(rows)
    dfC['log_n'] = np.log(dfC['n'])
    coef = np.polyfit(dfC['log_n'], dfC['Cmax_per_spin'], 1)
    return C_vs_T, dfC, coef

if __name__ == "__main__":
    import sys, os
    from time import time

    part = None
    for arg in sys.argv[1:]:
        if arg.startswith("--part="):
            part = arg.split("=")[1]
    if part is None:
        part = os.environ.get("PART", "1")

    print(f"Running Ising simulation for PART={part}")

    if part == "1":
        print("Running part 1: M(T) for n=50...")
        np.random.seed(0)
        J = 1.5
        print("Part 1: n=50 magnetization vs T")
        dfM = part1_magnetization_vs_T(n=50, J=J)

    elif part == "2":
        print("Running part 2: C(T) and scaling with log(n)...")
        print("\nPart 2: specific heat scaling")
        J = 1.5
        sizes = [5, 10, 20, 30, 40, 50]
        C_vs_T, dfC, coef = part2_specific_heat_sizes(sizes=sizes, J=J)
        print("\nC_max / N table:")
        print(dfC.to_string(index=False))
        print("\nLinear fit (Cmax/N) ~ a * log(n) + b  -> a,b = ", coef)

        plt.figure(figsize=(7, 5))
        for n in [5, 20, 50]:
            plt.plot(C_vs_T[n]['T'], C_vs_T[n]['C_per_spin'], marker='o', label=f'n={n}')
        plt.xlabel('T')
        plt.ylabel('C/N')
        plt.title('C/N vs T for sample sizes')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.plot(dfC['log_n'], dfC['Cmax_per_spin'], marker='o', linestyle='none')
        xs = np.linspace(dfC['log_n'].min() - 0.1, dfC['log_n'].max() + 0.1, 100)
        ys = coef[0] * xs + coef[1]
        plt.plot(xs, ys, label=f'fit: y={coef[0]:.3f} x + {coef[1]:.3f}')
        plt.xlabel('log(n)')
        plt.ylabel('C_max / N')
        plt.title('Finite-size scaling: C_max/N vs log(n)')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Unknown part; please choose PART=1 or PART=2.")



