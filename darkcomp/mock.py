"""Lightweight mock epoch astrometry for benchmarks (not a substitute for gaiamock).

Re-uses the real scan pattern (times, scan angles, parallax factors, per-CCD errors) of
template sources -- e.g. the 12 DR4 pre-release sources -- and injects a synthetic single-star
+ Keplerian signal plus Gaussian noise. The scanning law is therefore realistic for those sky
positions; gaiamock (El-Badry et al. 2024) should be used when the sky distribution matters.
"""
import numpy as np
import jax
import jax.numpy as jnp

from . import epoch as E, orbit


def random_population(rng, n, p_range=(30.0, 2500.0), a0_range=(0.05, 5.0), plx_range=(1.0, 10.0), e_max=0.8):
    """Dict of truth arrays for n sources. a0 in mas, periods log-uniform."""
    return dict(
        period=np.exp(rng.uniform(np.log(p_range[0]), np.log(p_range[1]), n)),
        ecc=rng.uniform(0.0, e_max, n),
        a0_mas=np.exp(rng.uniform(np.log(a0_range[0]), np.log(a0_range[1]), n)),
        inc=np.arccos(rng.uniform(-1, 1, n)), Omega=rng.uniform(0, 2 * np.pi, n), omega=rng.uniform(0, 2 * np.pi, n),
        t_peri=rng.uniform(-1000, 1000, n), parallax=rng.uniform(*plx_range, n),
        pmra=rng.normal(0, 20, n), pmdec=rng.normal(0, 20, n), d_ra_mas=rng.normal(0, 0.5, n), d_dec_mas=rng.normal(0, 0.5, n),
    )


def simulate(rng, templates, truth, noise_scale=1.0):
    """List of EpochAstrometry: source i uses templates[i % len(templates)]'s scan pattern."""
    out = []
    for i in range(len(truth["period"])):
        tpl = templates[i % len(templates)]
        keep = tpl.mask > 0
        t = tpl.t_day[keep]; psi = tpl.psi[keep]; f = tpl.plx_factor[keep]; err = tpl.err[keep] * noise_scale
        p = {k: v[i] for k, v in truth.items()}
        A, B, F, G = (float(x) for x in orbit.thiele_innes_ABFG(p["a0_mas"], p["omega"], p["Omega"], p["inc"]))
        X, Y = (np.asarray(x) for x in E.keplerian_xy(jnp.asarray(t), p["period"], p["ecc"], 2 * np.pi * p["t_peri"] / p["period"]))
        t_yr = t / E.YEAR_DAY
        ra = p["d_ra_mas"] + p["pmra"] * t_yr + B * X + G * Y
        dec = p["d_dec_mas"] + p["pmdec"] * t_yr + A * X + F * Y
        w = ra * np.sin(psi) + dec * np.cos(psi) + p["parallax"] * f + rng.normal(0, err)
        out.append(E.EpochAstrometry.from_arrays(t + tpl.ref_jd, w, err, f, np.degrees(psi), ref_jd=tpl.ref_jd))
    return out
