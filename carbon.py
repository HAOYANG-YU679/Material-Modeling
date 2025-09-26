#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='14C decay simulation')
parser.add_argument('--plot', type=int, default=10, help='Time-step width in years for numerical simulation')
args = parser.parse_args()
dt = args.plot

T_half = 5700
tau = T_half / np.log(2)
N0 = 4.3e13
t_max =20000

t_exact = np.linspace(0, t_max, 1000)
R_exact = N0 / tau * np.exp(-t_exact / tau)

t_num = np.arange(0, t_max+dt, dt)
N_num = np.zeros_like(t_num, dtype=float)
N_num[0] = N0
for i in range(1, len(t_num)):
    N_num[i] = N_num[i-1] * (1 - dt/tau)
R_num = N_num / tau

plt.figure(figsize=(8,5))
plt.plot(t_exact, R_exact, 'k-', label='Exact solution')
plt.plot(t_num, R_num, 'o-', label=f'Numerical dt={dt} yr')
plt.xlabel('Time [years]')
plt.ylabel('Activity [atoms/year]')
plt.title('14C Decay Activity: Numerical vs Exact')
plt.legend()
plt.savefig("carbon_plot.png", dpi=300)
plt.show()
