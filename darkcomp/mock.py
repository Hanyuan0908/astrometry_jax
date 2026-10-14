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


def physical_population(rng, n, m_star=1.0, m_2=1.4, period=None, dist_pc=500.0, p_range=(30.0, 2500.0),
                        e_max=0.8, eps=0.0):
    """Truth dict like `random_population`, but a0 follows from the physics of a dark companion.

    m_star, m_2 [Msun], dist_pc and period [d] may be scalars or length-n arrays; period=None draws
    log-uniform in p_range. parallax = 1000/dist_pc; a0 = a(P, M*+M2) * parallax * (q/(1+q) - eps/(1+eps))
    with q = M2/M* and eps the companion/primary flux ratio (0 for a dark companion). Orientation isotropic.
    """
    b = lambda x: np.broadcast_to(np.asarray(x, float), (n,)).copy()  # noqa: E731
    m_star, m_2, dist = b(m_star), b(m_2), b(dist_pc)
    P = b(period) if period is not None else np.exp(rng.uniform(np.log(p_range[0]), np.log(p_range[1]), n))
    plx = 1000.0 / dist
    a_au = np.asarray(orbit.semimajor_axis_au(P, m_star, m_2))
    a0 = np.asarray(orbit.photocentre_a0_mas(a_au, plx, m_2 / m_star, eps))
    return dict(
        period=P, ecc=rng.uniform(0.0, e_max, n), a0_mas=a0,
        inc=np.arccos(rng.uniform(-1, 1, n)), Omega=rng.uniform(0, 2 * np.pi, n), omega=rng.uniform(0, 2 * np.pi, n),
        t_peri=rng.uniform(-1000, 1000, n), parallax=plx,
        pmra=rng.normal(0, 20, n), pmdec=rng.normal(0, 20, n), d_ra_mas=rng.normal(0, 0.5, n), d_dec_mas=rng.normal(0, 0.5, n),
        m_star=m_star, m_2=m_2, dist_pc=dist,
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


def add_rvs(rng, epochs, truth, n_rv=10, err_kms=1.0, gamma_sigma=30.0, m_star=None):
    """Attach mock epoch RVs to each EpochAstrometry (dark companion: the photocentre is the star).

    n_rv epochs are drawn from the source's own transit times (RVS measures at the astrometric transits),
    K follows from a1 = a0 / parallax with the truth's (P, e, i), gamma ~ N(0, gamma_sigma). Adds
    truth["K"] and truth["gamma"]. Returns the new list.
    """
    out, Ks, gammas = [], [], []
    for i, ep in enumerate(epochs):
        p = {k: float(v[i]) for k, v in truth.items()}
        t_all = np.unique(np.round(ep.t_day[ep.mask > 0], 3))
        t = np.sort(rng.choice(t_all, size=min(n_rv, len(t_all)), replace=False))
        K = float(orbit.rv_semi_amplitude_kms(p["a0_mas"] / p["parallax"], p["period"], p["ecc"], p["inc"]))
        gamma = float(rng.normal(0.0, gamma_sigma))
        rv = np.asarray(orbit.radial_velocity_kms(jnp.asarray(t), p["period"], p["ecc"], p["t_peri"], p["omega"], K, gamma))
        rv = rv + rng.normal(0.0, err_kms, len(t))
        out.append(ep.with_rvs(t + ep.ref_jd, rv, np.full(len(t), err_kms)))
        Ks.append(K); gammas.append(gamma)
    truth["K"] = np.array(Ks); truth["gamma"] = np.array(gammas)
    return out
