#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description="Golf ball trajectory simulation")
parser.add_argument("--plot", type=float, default=15.0)

args = parser.parse_args()
theta_deg = args.plot

m = 0.046        
v0 = 70.0       
rho = 1.29       
A = 0.0014         
g = 9.81           
dt = 0.01          
S0_omega_over_m = 0.25

theta = np.radians(theta_deg)
vx, vy = v0*np.cos(theta), v0*np.sin(theta)
x, y = 0.0, 0.0

trajectories = {}

def simulate(case):
    global vx, vy, x, y
    x, y = 0.0, 0.0
    theta = np.radians(theta_deg)
    vx, vy = v0*np.cos(theta), v0*np.sin(theta)
    traj_x, traj_y = [x], [y]

    while y >= 0:
        v = np.sqrt(vx**2 + vy**2)

        Fx, Fy = 0.0, -m*g
        
        if case == "smooth":
            C = 0.5
            Fd = -C * rho * A * v * np.array([vx, vy])
            Fx += Fd[0]
            Fy += Fd[1]

        elif case == "dimpled":
            if v <= 14:
                C = 0.5
            else:
                C = 7.0 / v
            Fd = -C * rho * A * v * np.array([vx, vy])
            Fx += Fd[0]
            Fy += Fd[1]

        elif case == "spin":
            if v <= 14:
                C = 0.5
            else:
                C = 7.0 / v
            Fd = -C * rho * A * v * np.array([vx, vy])
            Fx += Fd[0]
            Fy += Fd[1]

            Fm_x = S0_omega_over_m * (-vy) * m
            Fm_y = S0_omega_over_m * (vx) * m
            Fx += Fm_x
            Fy += Fm_y    

        vx += Fx/m * dt
        vy += Fy/m * dt
        x += vx * dt
        y += vy * dt

        traj_x.append(x)
        traj_y.append(y)

    return np.array(traj_x), np.array(traj_y)

trajectories["ideal"] = simulate("ideal")
trajectories["smooth"] = simulate("smooth")
trajectories["dimpled"] = simulate("dimpled")
trajectories["spin"] = simulate("spin")

plt.figure(figsize=(8,5))
markers = {"ideal":"k-","smooth":"b--","dimpled":"g-.", "spin":"r:"}
for case, (tx, ty) in trajectories.items():
    plt.plot(tx, ty, markers[case], label=case)


plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"Golf Ball Trajectory (\u03B8={theta_deg}\u00B0)")
plt.legend()
plt.tight_layout()
plt.savefig("golf_plot.png", dpi=300)
plt.show()
