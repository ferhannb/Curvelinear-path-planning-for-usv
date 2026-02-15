import numpy as np
import casadi as ca


def sinc(x):
    return ca.if_else(ca.fabs(x) < 1e-8, 1 - x**2 / 6, ca.sin(x) / x)


def wrap_to_pi(a):
    return ca.atan2(ca.sin(a), ca.cos(a))


def step_constK_sinc(x, y, psi, K, ds):
    dpsi = K * ds
    fac = sinc(dpsi / (2.0 * np.pi))
    x1 = x + ds * fac * ca.cos(psi + dpsi / 2.0)
    y1 = y + ds * fac * ca.sin(psi + dpsi / 2.0)
    psi1 = psi + dpsi
    return x1, y1, psi1


def clothoid_increment_numeric(x0, y0, psi0, K0, K1, ds, nseg=16):
    ds_seg = ds / nseg
    x, y, psi = x0, y0, psi0
    for i in range(nseg):
        K_mid = K0 + (K1 - K0) * ((i + 0.5) / nseg)
        x, y, psi = step_constK_sinc(x, y, psi, K_mid, ds_seg)
    return x, y, psi


def K_next_fixed_ramp(Kcur, Kcmd, ds, K_MAX=0.30, S_MAX=14.0, eps=1e-9):
    a0 = K_MAX / S_MAX
    delta = Kcmd - Kcur
    max_step = a0 * ds
    dK = max_step * ca.tanh(delta / (max_step + eps))
    return Kcur + dK


def wrap_to_pi_np(a):
    return np.arctan2(np.sin(a), np.cos(a))


def step_constK_sinc_np(x, y, psi, K, ds):
    dpsi = K * ds
    fac = np.sinc(dpsi / (2.0 * np.pi))
    x1 = x + ds * fac * np.cos(psi + dpsi / 2.0)
    y1 = y + ds * fac * np.sin(psi + dpsi / 2.0)
    psi1 = psi + dpsi
    return x1, y1, psi1


def clothoid_increment_numeric_np(x0, y0, psi0, K0, K1, ds, nseg=16):
    ds_seg = ds / nseg
    x, y, psi = x0, y0, psi0
    for i in range(nseg):
        K_mid = K0 + (K1 - K0) * ((i + 0.5) / nseg)
        x, y, psi = step_constK_sinc_np(x, y, psi, K_mid, ds_seg)
    return x, y, psi


def K_next_fixed_ramp_np(Kcur, Kcmd_val, ds_val, K_MAX=0.30, S_MAX=14.0, eps=1e-9):
    a0 = K_MAX / S_MAX
    delta = Kcmd_val - Kcur
    max_step = a0 * ds_val
    dK = max_step * np.tanh(delta / (max_step + eps))
    return Kcur + dK
