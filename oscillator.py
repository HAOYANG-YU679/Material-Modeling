#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sin, cos, pi, atan2

try:
        from scipy.signal import find_peaks
except Exception:
        find_peaks = None

g = 9.8
l = 9.8
gamma = 0.25
alphaD_default = 0.2

omega0 = np.sqrt(g / l)



def euler_cromer(theta0, omega0_init, tmax, dt, OmegaD, alphaD, gamma, nonlinear=False):
    nsteps = int(tmax / dt)
    t = np.linspace(0, tmax, nsteps+1)
    theta = np.zeros_like(t)
    omega = np.zeros_like(t)
    theta[0] = theta0
    omega[0] = omega0_init
    for i in range(nsteps):
        driving = alphaD * np.sin(OmegaD * t[i])
        restoring = - (g/l) * np.sin(theta[i]) if nonlinear else - (g/l) * theta[i]
        omega[i+1] = omega[i] + dt*(restoring - 2*gamma*omega[i] + driving)
        theta[i+1] = theta[i] + dt*omega[i+1]
    return t, theta, omega

def rk4(theta0, omega0_init, tmax, dt, OmegaD, alphaD, gamma, nonlinear=False):
    nsteps = int(tmax / dt)
    t = np.linspace(0, tmax, nsteps+1)
    theta = np.zeros_like(t)
    omega = np.zeros_like(t)
    theta[0] = theta0
    omega[0] = omega0_init

    def derivs(ti, yi):
        th, om = yi
        restoring = - (g/l) * np.sin(th) if nonlinear else - (g/l) * th
        dth = om
        dom = restoring - 2*gamma*om + alphaD*np.sin(OmegaD*ti)
        return np.array([dth, dom])

    for i in range(nsteps):
        ti = t[i]
        yi = np.array([theta[i], omega[i]])
        k1 = derivs(ti, yi)
        k2 = derivs(ti + dt/2, yi + dt*k1/2)
        k3 = derivs(ti + dt/2, yi + dt*k2/2)
        k4 = derivs(ti + dt, yi + dt*k3)
        y_next = yi + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        theta[i+1], omega[i+1] = y_next
    return t, theta, omega

def compute_steady_amplitude_phase(OmegaD, alphaD, gamma, method='rk4', nonlinear=False,
                                   ttransient=200.0, tmeasure=100.0, dt=0.01):

    tmax = ttransient + tmeasure
    theta0_init = 0.01
    omega0_init = 0.0
    if method.lower().startswith('euler'):
        t, th, om = euler_cromer(theta0_init, omega0_init, tmax, dt, OmegaD, alphaD, gamma, nonlinear)
    else:
        t, th, om = rk4(theta0_init, omega0_init, tmax, dt, OmegaD, alphaD, gamma, nonlinear)

    idx = t >= (tmax - tmeasure)
    t_meas = t[idx]
    th_meas = th[idx]

    if find_peaks:
        peaks, _ = find_peaks(th_meas)
        troughs, _ = find_peaks(-th_meas)
        peak_vals = th_meas[peaks] if len(peaks)>0 else np.array([np.max(th_meas)])
        trough_vals = th_meas[troughs] if len(troughs)>0 else np.array([np.min(th_meas)])
        if peak_vals.size>0 and trough_vals.size>0:
            amp = 0.5*(np.mean(peak_vals) - np.mean(trough_vals))
        else:
            amp = 0.5*(np.max(th_meas)-np.min(th_meas))
    else:
        amp = 0.5*(np.max(th_meas)-np.min(th_meas))


    drive_sin = np.sin(OmegaD * t_meas)
    drive_cos = np.cos(OmegaD * t_meas)
    a = 2.0*np.mean(th_meas * drive_sin)
    b = 2.0*np.mean(th_meas * drive_cos)

    phi = atan2(b, a)
    return amp, phi

def part1_estimate_resonance():

    print("Small-angle natural frequency \u03C90 = sqrt(g/l) = {:.6f} s^-1".format(omega0))
    print("Resonant response will be near \u03C90; for weak damping peak is close to \u03C90 but slightly shifted.")

    print("Damping \u03B3 = {:.3f} s^-1; quality factor Q \u2248 \u03C90/(2\u03B3) = {:.3f}".format(gamma, omega0/(2*gamma)))

    return omega0

def part2_time_series_and_resonance(method='rk4', alphaD=alphaD_default, gamma_in=gamma):

    OmegaD = omega0
    dt = 0.01
    tmax = 200.0
    print(f"Integrating with method={method}, OmegaD={OmegaD:.4f}, alphaD={alphaD}, gamma={gamma_in}")
    if method.lower().startswith('euler'):
        t, th, om = euler_cromer(0.01, 0.0, tmax, dt, OmegaD, alphaD, gamma_in, nonlinear=False)
    else:
        t, th, om = rk4(0.01, 0.0, tmax, dt, OmegaD, alphaD, gamma_in, nonlinear=False)

    plt.figure(figsize=(8,5))
    plt.plot(t, th, label=r'$\theta(t)$')
    plt.plot(t, om, label=r'$\omega(t)$', alpha=0.7)
    plt.xlabel('t (s)')
    plt.legend()
    plt.title(f"Time series (OmegaD={OmegaD:.3f}, method={method})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('osc_time_series_part2.png', dpi=200)
    plt.close()

    Omegas = np.linspace(0.2*omega0, 2.0*omega0, 50)
    amps = []
    phis = []
    for Om in Omegas:
        amp, phi = compute_steady_amplitude_phase(Om, alphaD, gamma_in, method=method, nonlinear=False,
                                                  ttransient=200.0, tmeasure=50.0, dt=0.01)
        amps.append(amp)
        phis.append(phi)
    amps = np.array(amps)
    phis = np.array(phis)


    plt.figure()
    plt.plot(Omegas, amps, marker='o')
    plt.axvline(omega0, color='k', linestyle='--', label=r'$\omega_0$')
    plt.xlabel(r'$\Omega_D$ (s$^{-1}$)')
    plt.ylabel(r'Amplitude $\theta_0$ (rad)')
    plt.title('Resonance curve (Amplitude)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('osc_resonance_amplitude_part2.png', dpi=200)
    plt.close()

    plt.figure()
    plt.plot(Omegas, np.unwrap(phis), marker='s')
    plt.xlabel(r'$\Omega_D$ (s$^{-1}$)')
    plt.ylabel(r'Phase shift $\phi$ (rad)')
    plt.title('Resonance curve (Phase)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('osc_resonance_phase_part2.png', dpi=200)
    plt.close()

    maxA = amps.max()
    half = maxA / 2.0
    idxs = np.where(amps >= half)[0]
    if idxs.size > 0:
        fwhm = Omegas[idxs[-1]] - Omegas[idxs[0]]
    else:
        fwhm = np.nan
        print(f"Max amplitude = {maxA:.5f}; FWHM \u2248 {fwhm:.5f}. Compare to damping scale ~ \u03B3 = {gamma_in:.3f}")
    return Omegas, amps, phis

def part3_energy_plot(OmegaD=None, method='rk4'):
    if OmegaD is None:
        OmegaD = omega0
    tmax = 200.0
    dt = 0.001
    t, th, om = rk4(0.01, 0.0, tmax, dt, OmegaD, alphaD_default, gamma, nonlinear=False)

    T = 2*pi / OmegaD
    t_end = tmax
    t_start = t_end - 10*T
    idx = (t >= t_start) & (t <= t_end)
    te = t[idx]; the = th[idx]; oe = om[idx]

    m = 1.0
    U = m * g * l * (1 - np.cos(the))
    K = 0.5 * m * (l * oe)**2
    E = U + K
    plt.figure()
    plt.plot(te, U, label='Potential U')
    plt.plot(te, K, label='Kinetic K')
    plt.plot(te, E, label='Total E')
    plt.xlabel('t (s)')
    plt.legend()
    plt.title(f'Energies over ~10 periods (OmegaD={OmegaD:.3f})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('osc_energies_part3.png', dpi=200)
    plt.close()

def part4_nonlinear(alphaD_val=1.2, OmegaD_val=None, method='rk4'):

    if OmegaD_val is None:
        OmegaD_val = omega0
    dt = 0.01
    tmax = 200.0
    t, th_lin, om_lin = rk4(0.01, 0.0, tmax, dt, OmegaD_val, alphaD_val, gamma, nonlinear=False)
    t, th_non, om_non = rk4(0.01, 0.0, tmax, dt, OmegaD_val, alphaD_val, gamma, nonlinear=True)

    idx = t > (tmax - 50)
    plt.figure(figsize=(8,5))
    plt.plot(t[idx], th_lin[idx], label='linear')
    plt.plot(t[idx], th_non[idx], label='nonlinear', alpha=0.8)
    plt.xlabel('t (s)')
    plt.ylabel(r'$\theta$ (rad)')
    plt.legend()
    plt.title(f'Linear vs Nonlinear response (OmegaD={OmegaD_val:.3f}, alphaD={alphaD_val})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('osc_linear_vs_nonlinear_part4.png', dpi=200)
    plt.close()

def part5_lyapunov(alphaD_vals=[0.2,0.5,1.2], OmegaD_val=0.666, dtheta0=0.001, method='rk4'):
    dt = 0.01
    tmax = 200.0
    for alphaD_val in alphaD_vals:

        t, th1, om1 = rk4(0.01, 0.0, tmax, dt, OmegaD_val, alphaD_val, gamma, nonlinear=True)
        t, th2, om2 = rk4(0.01 + dtheta0, 0.0, tmax, dt, OmegaD_val, alphaD_val, gamma, nonlinear=True)
        delta = np.abs(th1 - th2)

        delta[delta==0] = 1e-16
        ln_delta = np.log(delta)

        idx = (t > (0.2*tmax)) & (t < (0.8*tmax))
        if np.sum(idx) < 10:
            idx = t>0
        coef = np.polyfit(t[idx], ln_delta[idx], 1)
        lam = coef[0]
        print(f"alphaD={alphaD_val}: estimated Lyapunov exponent \u03BB \u2248 {lam:.5f} (1/s)")
        plt.figure()
        plt.plot(t, ln_delta, label=r'$\ln|\Delta\theta(t)|$')
        plt.plot(t, coef[0]*t + coef[1], '--', label=f'linear fit slope={coef[0]:.4f}')
        plt.xlabel('t (s)')
        plt.ylabel(r'$\ln|\Delta\theta|$')
        plt.title(f'Lyapunov estimate alphaD={alphaD_val}, OmegaD={OmegaD_val}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'osc_lyapunov_alpha_{alphaD_val}.png', dpi=200)
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', type=int, default=1, help='Which part to run (1..5)')
    p.add_argument('--method', type=str, default='rk4', choices=['rk4','euler'], help='Integrator')
    p.add_argument('--alphaD', type=float, default=alphaD_default, help='Driving amplitude')
    p.add_argument('--OmegaD', type=float, default=None, help='Driving frequency (overrides default)')
    p.add_argument('--gamma', type=float, default=gamma, help='Damping gamma')
    args = p.parse_args()

    os.makedirs('figures', exist_ok=True)
    os.chdir('figures')

    if args.part == 1:
        w0 = part1_estimate_resonance()
        print(f"Estimated resonance near Omega_D \u2248 {w0:.6f} s^-1")
    elif args.part == 2:
        Omegas, amps, phis = part2_time_series_and_resonance(method=args.method, alphaD=args.alphaD, gamma_in=args.gamma)
        print("Saved: osc_time_series_part2.png, osc_resonance_amplitude_part2.png, osc_resonance_phase_part2.png")
    elif args.part == 3:
        part3_energy_plot(OmegaD=args.OmegaD)
        print("Saved: osc_energies_part3.png")
    elif args.part == 4:
        part4_nonlinear(alphaD_val=args.alphaD, OmegaD_val=(args.OmegaD if args.OmegaD else omega0), method=args.method)
        print("Saved: osc_linear_vs_nonlinear_part4.png")
    elif args.part == 5:
        part5_lyapunov(alphaD_vals=[0.2, 0.5, 1.2], OmegaD_val=(args.OmegaD if args.OmegaD else 0.666), dtheta0=0.001, method=args.method)
        print("Saved lyapunov plots: osc_lyapunov_*.png")
    else:
        print("Unknown part. Choose 1..5")

if __name__ == '__main__':
    main()
