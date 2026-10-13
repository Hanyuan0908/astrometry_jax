"""Keplerian orbit geometry: Campbell elements <-> Thiele-Innes, photocentre size, RV curve.

Conventions follow the Gaia DR3 NSS archive and El-Badry et al. 2024 eqs. 1-10:
  A =  a0 (cos w cos W - sin w sin W cos i)
  B =  a0 (cos w sin W + sin w cos W cos i)
  F = -a0 (sin w cos W + cos w sin W cos i)
  G = -a0 (sin w sin W - cos w cos W cos i)
  C =  a1 sin w sin i,  H = a1 cos w sin i
with A,B,F,G in mas (photocentre orbit) and C,H in au (orbit of the RV star).
Angles in radians everywhere inside the code; w = omega (argument of periastron of the
luminous star), W = Omega (longitude of ascending node), i = inclination in [0, pi].
"""
import jax.numpy as jnp
import numpy as np

from .constants import GM_SUN_AU3_DAY2, AU_PER_DAY_TO_KM_S, TWO_PI
from .kepler import true_anomaly, mean_anomaly


# ----------------------------------------------------------------------------- sizes

def semimajor_axis_au(period_day, m_star, m_2):
    """Relative semimajor axis a [au] from Kepler's third law (eq. 1)."""
    return (period_day**2 * GM_SUN_AU3_DAY2 * (m_star + m_2) / (4.0 * jnp.pi**2)) ** (1.0 / 3.0)


def a1_au(a_au, q):
    """Semimajor axis of the luminous star's orbit around the barycentre (eq. 2)."""
    return a_au * q / (1.0 + q)


def photocentre_a0_mas(a_au, parallax_mas, q, eps=0.0):
    """Angular photocentre semimajor axis (eq. 4). a/d in arcsec = a[au]*parallax[arcsec]."""
    return a_au * parallax_mas * (q / (1.0 + q) - eps / (1.0 + eps))


def rv_semi_amplitude_kms(a1_au_, period_day, e, inc):
    """K_star [km/s] (eq. 3)."""
    return TWO_PI * a1_au_ * jnp.sin(inc) / (period_day * jnp.sqrt(1.0 - e**2)) * AU_PER_DAY_TO_KM_S


# ----------------------------------------------------------------------------- Thiele-Innes

def thiele_innes_ABFG(a0, omega, Omega, inc):
    cw, sw = jnp.cos(omega), jnp.sin(omega)
    cW, sW = jnp.cos(Omega), jnp.sin(Omega)
    ci = jnp.cos(inc)
    A = a0 * (cw * cW - sw * sW * ci)
    B = a0 * (cw * sW + sw * cW * ci)
    F = -a0 * (sw * cW + cw * sW * ci)
    G = -a0 * (sw * sW - cw * cW * ci)
    return A, B, F, G


def thiele_innes_CH(a1, omega, inc):
    si = jnp.sin(inc)
    return a1 * jnp.sin(omega) * si, a1 * jnp.cos(omega) * si


def campbell_from_thiele_innes(A, B, F, G):
    """Invert A,B,F,G -> (a0, omega, Omega, inc). Works with numpy or jax arrays.

    Uses the identities
        A+G = a0 (1+cos i) cos(w+W),   B-F = a0 (1+cos i) sin(w+W)
        A-G = a0 (1-cos i) cos(w-W),  -B-F = a0 (1-cos i) sin(w-W)
    which avoid catastrophic cancellation. Returned i is in [0, pi]; (w, W) are
    returned with W in [0, pi) -- the astrometry-only solution is invariant under
    (w, W) -> (w+pi, W+pi), equivalently i -> -i; RVs break that degeneracy.
    """
    xp = jnp if any(isinstance(v, jnp.ndarray) for v in (A, B, F, G)) else np
    s_plus = xp.hypot(A + G, B - F)
    s_minus = xp.hypot(A - G, B + F)
    a0 = 0.5 * (s_plus + s_minus)
    cos_i = (s_plus - s_minus) / (s_plus + s_minus)
    inc = xp.arccos(xp.clip(cos_i, -1.0, 1.0))
    wpW = xp.arctan2(B - F, A + G)
    wmW = xp.arctan2(-B - F, A - G)
    omega = 0.5 * (wpW + wmW)
    Omega = 0.5 * (wpW - wmW)
    # fold Omega into [0, pi) carrying omega along so that ABFG are unchanged
    shift = xp.floor(Omega / xp.pi) * xp.pi
    Omega = Omega - shift
    omega = omega - shift
    omega = xp.mod(omega, 2.0 * xp.pi)
    return a0, omega, Omega, inc


# ----------------------------------------------------------------------------- RV curve

def radial_velocity_kms(t_day, period_day, e, t_peri_day, omega, K_kms, gamma_kms):
    """RV of the luminous star: gamma + K [cos(w + f) + e cos w]."""
    M = mean_anomaly(t_day, period_day, t_peri_day)
    f = true_anomaly(M, e)
    return gamma_kms + K_kms * (jnp.cos(omega + f) + e * jnp.cos(omega))


# ----------------------------------------------------------------------------- mass functions

def rv_mass_function(period_day, K_kms, e):
    """f(M2) = P K^3 (1-e^2)^{3/2} / (2 pi G)  in Msun."""
    K_au_day = K_kms / AU_PER_DAY_TO_KM_S
    return period_day * K_au_day**3 * (1.0 - e**2) ** 1.5 / (TWO_PI * GM_SUN_AU3_DAY2)


def astrometric_mass_function(a0_mas, parallax_mas, period_day):
    """(a0/parallax)^3 (P/yr)^-2 in Msun (eq. 12)."""
    return (a0_mas / parallax_mas) ** 3 * (period_day / 365.25) ** (-2.0)
