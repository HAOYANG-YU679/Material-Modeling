#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

def random_walk_2D(N_steps=100, N_walks=10_000):
    directions = np.array([[-1,0],[1,0],[0,-1],[0,1]])
    idx = np.random.randint(0, 4, size=(N_walks, N_steps))
    steps = directions[idx]  # shape: (N_walks, N_steps, 2)
    positions = np.cumsum(steps, axis=1)
    return positions

def part1():
    N_steps = 100
    N_walks = 10_000
    positions = random_walk_2D(N_steps, N_walks)
    x = positions[:,:,0]

    n = np.arange(1, N_steps+1)
    mean_x = np.mean(x, axis=0)
    mean_x2 = np.mean(x**2, axis=0)

    plt.figure()
    plt.plot(n, mean_x, label=r'$\langle x_n \rangle$')
    plt.plot(n, mean_x2, label=r'$\langle x_n^2 \rangle$')
    plt.xlabel('n (steps)')
    plt.ylabel('Average')
    plt.title('2D Random Walk: Mean and Mean-Square of x')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('rwalk_part1.png', dpi=150)
    plt.close()
    print("Saved: rwalk_part1.png")

def part2():
    N_steps = 100
    N_walks = 10_000
    positions = random_walk_2D(N_steps, N_walks)

    x, y = positions[:,:,0], positions[:,:,1]
    r2 = x**2 + y**2

    n = np.arange(1, N_steps+1)
    mean_r2 = np.mean(r2, axis=0)

    coeffs = np.polyfit(n, mean_r2, 1)
    D_est = coeffs[0] / 4

    plt.figure()
    plt.plot(n, mean_r2, 'o', markersize=4, label=r'$\langle r^2 \rangle$ (data)')
    plt.plot(n, np.polyval(coeffs, n), 'r-', label=f'Linear fit (D ≈ {D_est:.3f})')
    plt.xlabel('Steps (t)')
    plt.ylabel(r'$\langle r^2 \rangle$')
    plt.title('2D Random Walk: Diffusive Behavior')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('rwalk_part2.png', dpi=150)
    plt.close()
    print(f"Saved: rwalk_part2.png\nEstimated diffusion constant D ≈ {D_est:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Random Walk simulation")
    parser.add_argument("--part", type=int, required=True, help="Part number (1 or 2)")
    args = parser.parse_args()

    if args.part == 1:
        part1()
    elif args.part == 2:
        part2()
    else:
        print("Error: --part must be 1 or 2.")
