"""Lightweight mock epoch astrometry for benchmarks (not a substitute for gaiamock).

Re-uses the real scan pattern (times, scan angles, parallax factors, per-CCD errors) of
template sources -- e.g. the 12 DR4 pre-release sources -- and injects a synthetic single-star
+ Keplerian signal plus Gaussian noise. The scanning law is therefore realistic for those sky
positions; gaiamock (El-Badry et al. 2024) should be used when the sky distribution matters.
"""
import json
import numpy as np

from . import epoch as E, orbit
from .constants import TWO_PI, AU_PER_DAY_TO_KM_S


# numpy (not JAX) orbit kinematics: the generators loop over sources in Python, and one un-jitted JAX call
# per source costs 15 ms on a laptop and far more on a GPU runtime (every op is dispatched to the device),
# which made the mock generation for an 8,000-source benchmark take longer than the search itself.
def _eccentric_anomaly(M, e, n_newton=20):
    M = np.mod(np.asarray(M, float), TWO_PI)
    Ea = M + 0.85 * e * np.sign(np.sin(M))                      # same starter as kepler.py
    for _ in range(n_newton):
        Ea = Ea - (Ea - e * np.sin(Ea) - M) / (1.0 - e * np.cos(Ea))
    return Ea


def keplerian_xy(t_day, period, ecc, phase):
    """numpy twin of `epoch.keplerian_xy` (agrees to 1e-12)."""
    Ea = _eccentric_anomaly(TWO_PI * np.asarray(t_day, float) / period - phase, ecc)
    return np.cos(Ea) - ecc, np.sqrt(1.0 - ecc**2) * np.sin(Ea)


def radial_velocity_kms(t_day, period, ecc, t_peri, omega, K, gamma):
    """numpy twin of `orbit.radial_velocity_kms`."""
    Ea = _eccentric_anomaly(TWO_PI * (np.asarray(t_day, float) - t_peri) / period, ecc)
    f = 2.0 * np.arctan2(np.sqrt(1.0 + ecc) * np.sin(Ea / 2.0), np.sqrt(1.0 - ecc) * np.cos(Ea / 2.0))
    return gamma + K * (np.cos(omega + f) + ecc * np.cos(omega))


def thiele_innes_ABFG(a0, omega, Omega, inc):
    cw, sw, cW, sW, ci = np.cos(omega), np.sin(omega), np.cos(Omega), np.sin(Omega), np.cos(inc)
    return (a0 * (cw * cW - sw * sW * ci), a0 * (cw * sW + sw * cW * ci),
            -a0 * (sw * cW + cw * sW * ci), -a0 * (sw * sW - cw * cW * ci))


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
        A, B, F, G = thiele_innes_ABFG(p["a0_mas"], p["omega"], p["Omega"], p["inc"])
        X, Y = keplerian_xy(t, p["period"], p["ecc"], TWO_PI * p["t_peri"] / p["period"])
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
        K = TWO_PI * (p["a0_mas"] / p["parallax"]) * np.sin(p["inc"]) / (p["period"] * np.sqrt(1.0 - p["ecc"]**2)) * AU_PER_DAY_TO_KM_S
        gamma = float(rng.normal(0.0, gamma_sigma))
        rv = radial_velocity_kms(t, p["period"], p["ecc"], p["t_peri"], p["omega"], K, gamma)
        rv = rv + rng.normal(0.0, err_kms, len(t))
        out.append(ep.with_rvs(t + ep.ref_jd, rv, np.full(len(t), err_kms)))
        Ks.append(K); gammas.append(gamma)
    truth["K"] = np.array(Ks); truth["gamma"] = np.array(gammas)
    return out


# ----------------------------------------------------------------------------- mock bank (generate once, reuse)

def make_bank(rng, templates, truth, n_rv=0, err_kms=1.0, noise_scale=1.0, meta=None):
    """Simulate every source of `truth` on the templates and pack the result compactly for `save_bank`.

    The scan pattern (times, angles, parallax factors, errors, mask) is stored once per template and the
    per-source data are only the noisy abscissae (float32), the epoch RVs (if n_rv > 0) and the truth, so
    10,000 sources take ~35 MB. `load_bank` rebuilds the stacked pytree that `batch.stack_epochs` would give.
    """
    n = len(truth["period"]); n_tpl = len(templates)
    n_max = max(len(t.t_day) for t in templates)
    sims = simulate(rng, templates, truth, noise_scale)
    if n_rv > 0:
        sims = add_rvs(rng, sims, truth, n_rv=n_rv, err_kms=err_kms)
    tpl = [E.EpochAstrometry.from_arrays(t.t_day + t.ref_jd, t.w, t.err * noise_scale, t.plx_factor, np.degrees(t.psi),
                                         ref_jd=t.ref_jd, n_pad=n_max) for t in templates]
    bank = dict(tpl_t_day=np.stack([t.t_day for t in tpl]), tpl_err=np.stack([t.err for t in tpl]),
                tpl_plx_factor=np.stack([t.plx_factor for t in tpl]), tpl_psi_deg=np.stack([np.degrees(t.psi) for t in tpl]),
                tpl_mask=np.stack([t.mask for t in tpl]).astype(np.int8), tpl_ref_jd=np.array([t.ref_jd for t in tpl]),
                tpl_idx=(np.arange(n) % n_tpl).astype(np.int16), w=np.zeros((n, n_max), np.float32))
    for i, s in enumerate(sims):
        bank["w"][i, :len(s.w)] = s.w
    if n_rv > 0:
        rv = [s.with_rvs(s.rv_t_day[s.rv_mask > 0] + s.ref_jd, s.rv[s.rv_mask > 0], s.rv_err[s.rv_mask > 0], n_pad=n_rv) for s in sims]
        for k in ("rv_t_day", "rv", "rv_err", "rv_mask"):
            bank[k] = np.stack([getattr(s, k) for s in rv])
    for k, v in truth.items():
        bank["truth_" + k] = np.asarray(v)
    bank["meta"] = np.array(json.dumps(meta or {}))
    return bank


def save_bank(path, bank):
    np.savez_compressed(path, **bank)


def load_bank(path, n=None, with_rvs=False, offset=0):
    """(dd, truth) for sources [offset, offset + n) of a bank written by `save_bank`.

    dd is the stacked pytree `batch.run_pipeline` takes (same keys as `batch.stack_epochs`; dtypes follow
    the global precision). with_rvs=False drops the RV rows so the same sources run astrometry-only.
    """
    import jax.numpy as jnp
    z = np.load(path)
    n_all = len(z["tpl_idx"]); n = n_all - offset if n is None else n
    if offset + n > n_all:
        raise ValueError(f"bank {path} holds {n_all} sources, asked for {offset}+{n}")
    sl = slice(offset, offset + n); idx = z["tpl_idx"][sl]
    t_day = z["tpl_t_day"][idx]; psi = np.radians(z["tpl_psi_deg"][idx])
    dd = dict(t_day=t_day, t_yr=t_day / E.YEAR_DAY, w=np.asarray(z["w"][sl], float), err=z["tpl_err"][idx],
              plx_factor=z["tpl_plx_factor"][idx], sin_psi=np.sin(psi), cos_psi=np.cos(psi), mask=np.asarray(z["tpl_mask"][idx], float))
    if with_rvs:
        if "rv" not in z.files:
            raise ValueError(f"bank {path} has no epoch RVs")
        dd.update({k: np.asarray(z[k][sl], float) for k in ("rv_t_day", "rv", "rv_err", "rv_mask")})
    dd = {k: jnp.asarray(v) for k, v in dd.items()}
    truth = {k[6:]: z[k][sl] for k in z.files if k.startswith("truth_")}
    return dd, truth


def bank_info(path):
    z = np.load(path)
    return dict(n_sources=len(z["tpl_idx"]), n_templates=len(z["tpl_ref_jd"]), n_obs_max=z["w"].shape[1],
                n_rv=(z["rv"].shape[1] if "rv" in z.files else 0), meta=json.loads(str(z["meta"])))
