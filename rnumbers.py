#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def uniform_distribution():
    Ns = [1000, 1_000_000]
    bins_list = [10, 20, 50, 100]

    for N in Ns:
        x = np.random.rand(N)
        for bins in bins_list:
            plt.figure()
            plt.hist(x, bins=bins, density=True, alpha=0.7, edgecolor='black')
            plt.title(f'Uniform distribution: N={N}, bins={bins}')
            plt.xlabel('x')
            plt.ylabel('Probability Density')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(f'uniform_N{N}_bins{bins}.png', dpi=150)
            plt.close()
            print(f'Saved: uniform_N{N}_bins{bins}.png')

def gaussian_distribution():
    def box_muller(n):
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)
        z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        return z  # mean=0, sigma=1

    Ns = [1000, 1_000_000]
    bins_list = [10, 20, 50, 100]

    for N in Ns:
        x = box_muller(N)
        for bins in bins_list:
            plt.figure()
            plt.hist(x, bins=bins, density=True, alpha=0.7, edgecolor='black', label='Generated')
            # overlay theoretical Gaussian
            x_grid = np.linspace(-5, 5, 400)
            plt.plot(x_grid, norm.pdf(x_grid, 0, 1), 'r-', label='Theoretical Gaussian')
            plt.title(f'Gaussian distribution (Box-Muller): N={N}, bins={bins}')
            plt.xlabel('x')
            plt.ylabel('Probability Density')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(f'gaussian_N{N}_bins{bins}.png', dpi=150)
            plt.close()
            print(f'Saved: gaussian_N{N}_bins{bins}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random number generation experiments")
    parser.add_argument("--part", type=int, required=True, help="Part number (1 or 2)")
    args = parser.parse_args()

    if args.part == 1:
        uniform_distribution()
    elif args.part == 2:
        gaussian_distribution()
    else:
        print("Error: --part must be 1 or 2.")
