"""Likelihood for Gaia epoch (along-scan) astrometry, as will be published in DR4.

Data model (Lindegren et al. 2012 Sect. 5.1; Halbwachs et al. 2023; Holl et al. 2023), one
1-D measurement w per CCD transit with scan position angle psi and parallax factor f:

    w = (d_ra* + pmra t) sin psi + (d_dec + pmdec t) cos psi + parallax f          (single star)
      + (B X + G Y) sin psi + (A X + F Y) cos psi                                     (+ Keplerian)

with X = cos E - e, Y = sqrt(1-e^2) sin E, E the eccentric anomaly, t in years from the
reference epoch, and A,B,F,G the Thiele-Innes constants of the photocentre orbit in mas.
Given the three nonlinear parameters (P, e, phase) the model is linear in nine parameters,
so the profile likelihood is one weighted least-squares solve. Epoch radial velocities, when
present, join the same solve: v = gamma + c1 (cos f + e) - c2 sin f with c1 = K cos omega,
c2 = K sin omega is linear too (Gaia's AstroSpectroSB1 C, H), so the joint model has 12 linear
parameters and the search cost barely changes; the physical link between K and the astrometric
orbit is imposed later, in the 14-parameter `log_prob` used for NUTS; `profile_loglike` does that
for a batch of trial (P, e, phase) with `vmap`, which is the GPU-friendly period search.

Everything here uses the reference-epoch-relative time `t_day` stored in the data pytree;
for DR4 the reference epoch is J2017.5 = JD 2457936.875 (TCB).
"""
import dataclasses
from dataclasses import dataclass
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from . import orbit, likelihood as L
from .kepler import eccentric_anomaly
from .constants import GAIA_REF_JD

DR4_REF_JD = 2457936.875   # J2017.5 TCB
YEAR_DAY = 365.25

LINEAR_NAMES = ["d_ra_mas", "d_dec_mas", "parallax", "pmra", "pmdec", "B", "G", "A", "F"]
RV_NAMES = ["gamma", "c1", "c2"]   # c1 = K cos omega, c2 = K sin omega [km/s]


# ----------------------------------------------------------------------------- data

@dataclass
class EpochAstrometry:
    t_day: np.ndarray      # days since ref_jd
    w: np.ndarray          # along-scan abscissa [mas]
    err: np.ndarray        # [mas]
    plx_factor: np.ndarray
    psi: np.ndarray        # scan position angle [rad]
    mask: np.ndarray       # 1 real, 0 padding
    ref_jd: float = DR4_REF_JD
    transit_id: np.ndarray = None
    rv_t_day: np.ndarray = None    # optional epoch radial velocities: days since ref_jd
    rv: np.ndarray = None          # km/s
    rv_err: np.ndarray = None      # km/s
    rv_mask: np.ndarray = None     # 1 real, 0 padding

    def with_rvs(self, jd, rv, err, n_pad=None):
        """Copy with epoch RVs attached (JD, km/s); padded to n_pad rows with mask 0."""
        jd, rv, err = (np.asarray(x, float) for x in (jd, rv, err))
        n = len(jd); n_pad = n if n_pad is None else max(n, n_pad)
        pad = lambda x, fill: np.concatenate([x, np.full(n_pad - n, fill)])  # noqa: E731
        return dataclasses.replace(self, rv_t_day=pad(jd - self.ref_jd, 0.0), rv=pad(rv, 0.0), rv_err=pad(err, 1.0),
                                   rv_mask=pad(np.ones(n), 0.0))

    @property
    def n_rv(self):
        return 0 if self.rv is None else int(self.rv_mask.sum())

    @classmethod
    def from_arrays(cls, jd, w, err, plx_factor, psi_deg, ref_jd=DR4_REF_JD, n_pad=None, transit_id=None):
        jd, w, err, plx_factor, psi_deg = (np.asarray(x, float) for x in (jd, w, err, plx_factor, psi_deg))
        n = len(jd); n_pad = n if n_pad is None else max(n, n_pad)
        pad = lambda x, fill: np.concatenate([x, np.full(n_pad - n, fill)])  # noqa: E731
        mask = pad(np.ones(n), 0.0)
        return cls(pad(jd - ref_jd, 0.0), pad(w, 0.0), pad(err, 1.0), pad(plx_factor, 0.0),
                   pad(np.radians(psi_deg), 0.0), mask, ref_jd,
                   None if transit_id is None else np.asarray(transit_id))

    @classmethod
    def from_bh3_table(cls, path, drop_outliers=True, **kw):
        """CDS table B.1 / esa/gaia-bhthree epochast.dat format."""
        cols = "transit_id ccd_id obs_time_tcb centroid_pos_al centroid_pos_error_al parallax_factor_al scan_pos_angle outlier_flag".split()
        df = pd.read_csv(path, names=cols, sep=r"\s+", comment="#")
        if drop_outliers:
            df = df[df.outlier_flag != 1]
        return cls.from_arrays(df.obs_time_tcb, df.centroid_pos_al, df.centroid_pos_error_al,
                               df.parallax_factor_al, df.scan_pos_angle, transit_id=df.transit_id.values, **kw)

    @property
    def n(self):
        return int(self.mask.sum())

    def as_jax(self):
        t_yr = self.t_day / YEAR_DAY
        d = dict(t_day=jnp.asarray(self.t_day), t_yr=jnp.asarray(t_yr), w=jnp.asarray(self.w),
                 err=jnp.asarray(self.err), plx_factor=jnp.asarray(self.plx_factor),
                 sin_psi=jnp.asarray(np.sin(self.psi)), cos_psi=jnp.asarray(np.cos(self.psi)),
                 mask=jnp.asarray(self.mask))
        if self.rv is not None:
            d.update(rv_t_day=jnp.asarray(self.rv_t_day), rv=jnp.asarray(self.rv), rv_err=jnp.asarray(self.rv_err),
                     rv_mask=jnp.asarray(self.rv_mask))
        return d


# ----------------------------------------------------------------------------- linear model

def single_star_design(d):
    """(n, 5) design matrix for d_ra*, d_dec, parallax, pmra, pmdec."""
    return jnp.stack([d["sin_psi"], d["cos_psi"], d["plx_factor"],
                      d["t_yr"] * d["sin_psi"], d["t_yr"] * d["cos_psi"]], axis=-1)


def keplerian_xy(t_day, period, ecc, phase):
    """Elliptical rectangular coordinates X, Y for mean anomaly 2 pi t / P - phase."""
    M = 2.0 * jnp.pi * t_day / period - phase
    E = eccentric_anomaly(M, ecc)
    return jnp.cos(E) - ecc, jnp.sqrt(1.0 - ecc**2) * jnp.sin(E)


def orbital_design(d, period, ecc, phase):
    """(n, 9) design matrix: 5 single-star columns + [X sin, Y sin, X cos, Y cos] for B, G, A, F."""
    X, Y = keplerian_xy(d["t_day"], period, ecc, phase)
    kep = jnp.stack([X * d["sin_psi"], Y * d["sin_psi"], X * d["cos_psi"], Y * d["cos_psi"]], axis=-1)
    return jnp.concatenate([single_star_design(d), kep], axis=-1)


def has_rv(d):
    """True when the data pytree carries epoch RVs (a static, key-based test: safe under jit)."""
    return "rv" in d


def astro_only(d):
    return {k: v for k, v in d.items() if not k.startswith("rv")}


def rv_design(d, period, ecc, phase):
    """(n_rv, 3) design matrix for (gamma, c1, c2): v = gamma + c1 (cos f + e) - c2 sin f."""
    X, Y = keplerian_xy(d["rv_t_day"], period, ecc, phase)
    den = 1.0 - ecc * X - ecc**2                       # = 1 - e cos E
    return jnp.stack([jnp.ones_like(X), X / den + ecc, -Y / den], axis=-1)


def joint_design(d, period, ecc, phase):
    """Astrometric (n, 9) design, extended to (n + n_rv, 12) block form when the data carry epoch RVs."""
    M = orbital_design(d, period, ecc, phase)
    if not has_rv(d):
        return M
    R = rv_design(d, period, ecc, phase)
    top = jnp.concatenate([M, jnp.zeros((M.shape[0], 3), M.dtype)], axis=1)
    bot = jnp.concatenate([jnp.zeros((R.shape[0], 9), M.dtype), R], axis=1)
    return jnp.concatenate([top, bot], axis=0)


def observations(d, jitter=0.0, rv_jitter=0.0):
    """(y, weight): abscissae then RVs; weight = mask / (err^2 + jitter^2), 0 for padding."""
    y, wgt = d["w"], d["mask"] / (d["err"] ** 2 + jitter**2)
    if has_rv(d):
        y = jnp.concatenate([y, d["rv"]]); wgt = jnp.concatenate([wgt, d["rv_mask"] / (d["rv_err"] ** 2 + rv_jitter**2)])
    return y, wgt


def whiten(M, d, jitter=0.0, rv_jitter=0.0):
    """Whitened design matrix and data. An astrometry-only design (n rows) on data that carry RVs uses
    the first n observations only, so single-star and astrometric fits keep working on joint data."""
    y, wgt = observations(d, jitter, rv_jitter)
    if M.shape[0] != y.shape[0]:
        y, wgt = y[:M.shape[0]], wgt[:M.shape[0]]
    sw = jnp.sqrt(wgt)
    return M * sw[:, None], y * sw


def rv_constant_chi2(d):
    """chi2 of the RV data about their weighted mean (the 'single star' RV model); 0 without RVs."""
    if not has_rv(d):
        return jnp.zeros(())
    wgt = d["rv_mask"] / d["rv_err"] ** 2
    mean = jnp.sum(wgt * d["rv"]) / jnp.maximum(jnp.sum(wgt), 1e-30)
    return jnp.sum(wgt * (d["rv"] - mean) ** 2)


def weighted_lstsq(M, d, jitter=0.0, rv_jitter=0.0):
    """Weighted linear least squares: column-equilibrated normal equations solved by Cholesky.

    Returns (beta, chi2, cov). Column scaling keeps the normal matrix well conditioned enough
    for float32 (checked against a QR solve: log-likelihood differences ~0.02 in float32,
    1e-11 in float64) at ~1.5x the speed of QR. Padded rows have mask 0 -> zero weight.
    """
    Mw, yw = whiten(M, d, jitter, rv_jitter)
    scale = jnp.sqrt(jnp.sum(Mw**2, axis=0))
    dead = scale <= 0                                   # columns with no data (e.g. RV terms of a source without RVs)
    scale = jnp.where(dead, 1.0, scale)
    Ms = Mw / scale
    N = Ms.T @ Ms + jnp.diag(dead.astype(Ms.dtype))     # dead parameters are pinned to 0 with unit (meaningless) variance
    c, low = jax.scipy.linalg.cho_factor(N)
    bs = jax.scipy.linalg.cho_solve((c, low), Ms.T @ yw)
    r = yw - Ms @ bs
    chi2 = jnp.sum(r**2)
    Ninv = jax.scipy.linalg.cho_solve((c, low), jnp.eye(N.shape[0], dtype=N.dtype))
    return bs / scale, chi2, Ninv / jnp.outer(scale, scale)


def fit_single_star(d, jitter=0.0):
    return weighted_lstsq(single_star_design(d), d, jitter)


def profile_loglike(period, ecc, phase, d, jitter=0.0):
    """-chi2/2 after profiling the 9 (12 with RVs) linear parameters out at fixed (P, e, phase)."""
    _, chi2, _ = weighted_lstsq(joint_design(d, period, ecc, phase), d, jitter)
    return -0.5 * chi2


def profile_loglike_batch(P, E, PH, d, jitter=0.0, chunk=256):
    """Profile log-likelihood for flat arrays of trial (P, e, phase), evaluated in chunks.

    Each chunk materialises `chunk` copies of the (n_obs, 9) design matrix. Keep
    chunk * n_obs * 9 * 8 bytes small (default 256 -> ~11 MB for 600 observations); the
    whole grid is NEVER vmapped at once.
    """
    n = P.shape[0]
    n_pad = (-n) % chunk
    P = jnp.concatenate([P, jnp.full(n_pad, P[0])]); E = jnp.concatenate([E, jnp.full(n_pad, E[0])])
    PH = jnp.concatenate([PH, jnp.full(n_pad, PH[0])])
    f = jax.vmap(lambda p, e, ph: profile_loglike(p, e, ph, d, jitter))
    out = jax.lax.map(lambda x: f(*x), (P.reshape(-1, chunk), E.reshape(-1, chunk), PH.reshape(-1, chunk)))
    return out.reshape(-1)[:n]


def period_search(d, periods, n_ecc=4, n_phase=8, jitter=0.0, e_max=0.6, chunk=256, max_points=200_000):
    """Profile-likelihood periodogram: for each trial period, best -chi2/2 over an (e, phase) grid.

    Memory-safe: the len(periods)*n_ecc*n_phase grid is evaluated `chunk` points at a time.
    Returns (best_loglike_per_period, best_e, best_phase).
    """
    periods = jnp.asarray(periods)
    n_grid = len(periods) * n_ecc * n_phase
    if n_grid > max_points:
        raise ValueError(f"grid of {n_grid} points exceeds max_points={max_points}; coarsen the grid")
    eg = jnp.linspace(0.0, e_max, n_ecc)
    pg = jnp.linspace(0.0, 2 * jnp.pi, n_phase, endpoint=False)
    P, E, PH = jnp.meshgrid(periods, eg, pg, indexing="ij")
    ll = profile_loglike_batch(P.ravel(), E.ravel(), PH.ravel(), d, jitter, chunk).reshape(P.shape)
    flat = ll.reshape(len(periods), -1)
    k = jnp.argmax(flat, axis=1)
    idx = jnp.arange(len(periods))
    return flat[idx, k], E.reshape(len(periods), -1)[idx, k], PH.reshape(len(periods), -1)[idx, k]


def profile_residuals(x, d, jitter=0.0):
    """Whitened residual vector after profiling the linear parameters; x = (log P, e, phase)."""
    M = joint_design(d, jnp.exp(x[0]), x[1], x[2])
    Mw, yw = whiten(M, d, jitter)
    scale = jnp.sqrt(jnp.sum(Mw**2, axis=0)); dead = scale <= 0; scale = jnp.where(dead, 1.0, scale); Ms = Mw / scale
    c, low = jax.scipy.linalg.cho_factor(Ms.T @ Ms + jnp.diag(dead.astype(Ms.dtype)))
    bs = jax.scipy.linalg.cho_solve((c, low), Ms.T @ yw)
    return yw - Ms @ bs


def lm_refine(d, x0, jitter=0.0, n_iter=30, lo=None, hi=None, lam0=1e-2):
    """Levenberg-Marquardt on x = (log P, e, phase); see `lm.levenberg_marquardt`."""
    from .lm import levenberg_marquardt
    lo = jnp.array([jnp.log(1.0), 0.0, -1e3]) if lo is None else lo
    hi = jnp.array([jnp.log(1e5), 0.95, 1e3]) if hi is None else hi
    return levenberg_marquardt(lambda x: profile_residuals(x, d, jitter), x0, n_iter, lo, hi, lam0)


def refine_profile(d, period, ecc, phase, jitter=0.0, fit_jitter=False, n_iter=30, n_jitter_iter=6):
    """Polish (P, e, phase) with Levenberg-Marquardt; optionally estimate an excess-noise term.

    With fit_jitter=True a white-noise term (added in quadrature to every error, like Gaia's
    astrometric_excess_noise) is iterated so that chi2/dof -> 1: fit at fixed jitter, update
    jitter^2 from the residual excess, repeat. Needed for bright stars (e.g. HD 114762) whose
    per-CCD errors are underestimated. Returns Campbell elements, linear params and covariance.
    """
    x = jnp.array([jnp.log(period), ecc, phase])
    jit = float(jitter)
    for _ in range(n_jitter_iter if fit_jitter else 1):
        x, chi2 = lm_refine(d, x, jit, n_iter)
        if fit_jitter:
            n = float(d["mask"].sum()); dof = n - 12
            # excess variance needed to bring the astrometric chi2 to dof (RV rows are left alone)
            Mj = joint_design(d, jnp.exp(x[0]), x[1], x[2]); yj, _ = observations(d)
            n_ast = d["w"].shape[0]
            r_raw = np.asarray(yj - Mj @ weighted_lstsq(Mj, d, jit)[0])[:n_ast]
            err2 = np.asarray(d["err"]) ** 2; m = np.asarray(d["mask"]) > 0
            # solve sum (r^2 / (err^2 + j^2)) = dof for j^2 by bisection on log j
            f = lambda j2: np.sum(r_raw[m] ** 2 / (err2[m] + j2)) - dof  # noqa: E731
            if f(0.0) <= 0:
                jit = 0.0
            else:
                a, b = 1e-8, 1e4
                for _ in range(60):
                    mid = np.sqrt(a * b)
                    a, b = (mid, b) if f(mid) > 0 else (a, mid)
                jit = float(np.sqrt(a))
    P, e, ph = float(jnp.exp(x[0])), float(x[1]), float(x[2])
    Mj = joint_design(d, P, e, ph)
    beta, chi2, cov = weighted_lstsq(Mj, d, jit)
    beta = np.asarray(beta); names = LINEAR_NAMES + (RV_NAMES if has_rv(d) else [])
    lin = dict(zip(names, beta))
    a0, omega, Omega, inc = orbit.campbell_from_thiele_innes(lin["A"], lin["B"], lin["F"], lin["G"])
    t_peri = ph / (2 * np.pi) * P   # M = 2 pi (t - t_peri)/P  with phase = 2 pi t_peri / P
    ti = np.array([lin[k] for k in ("B", "G", "A", "F")]); cti = np.asarray(cov)[5:9, 5:9]
    significance = float(np.sqrt(ti @ np.linalg.solve(cti, ti)))
    out = dict(period=P, ecc=e, phase=ph, t_peri=t_peri, a0_mas=a0, omega=omega, Omega=Omega, inc=inc,
               chi2=float(chi2), jitter=jit, significance=significance, linear=lin,
               linear_cov=np.asarray(cov), n=int(d["mask"].sum()))
    if has_rv(d):
        # RV amplitude and argument of periastron from (c1, c2); the RV sign fixes the astrometric branch:
        # (omega, Omega) and (omega + pi, Omega + pi) give the same abscissae but opposite RV curves
        K = float(np.hypot(lin["c1"], lin["c2"])); omega_rv = float(np.arctan2(lin["c2"], lin["c1"]))
        if np.cos(omega_rv - omega) < 0:
            omega = (omega + np.pi) % (2 * np.pi); Omega = (Omega + np.pi) % (2 * np.pi)
        yj, wj = observations(d, jit); r = np.asarray((yj - Mj @ beta) ** 2 * wj)
        out.update(omega=omega, Omega=Omega, gamma=float(lin["gamma"]), K=K, omega_rv=omega_rv,
                   K_pred=float(orbit.rv_semi_amplitude_kms(a0 / lin["parallax"], P, e, inc)),   # dark companion
                   chi2_rv=float(r[d["w"].shape[0]:].sum()), n_rv=int(d["rv_mask"].sum()))
    return out


# ----------------------------------------------------------------------------- physical-parameter likelihood

def predict_abscissa(theta, d):
    """Along-scan model from the 14 physical parameters of `likelihood.PARAM_NAMES`."""
    p = L.derived(theta)
    X, Y = keplerian_xy(d["t_day"], p["period"], p["ecc"], 2.0 * jnp.pi * p["t_peri"] / p["period"])
    ra = p["d_ra_mas"] + p["pmra"] * d["t_yr"] + p["B"] * X + p["G"] * Y
    dec = p["d_dec_mas"] + p["pmdec"] * d["t_yr"] + p["A"] * X + p["F"] * Y
    return ra * d["sin_psi"] + dec * d["cos_psi"] + p["parallax"] * d["plx_factor"]


def log_like_epoch(theta, data, jitter=0.0):
    d = data["epoch"]
    var = d["err"] ** 2 + jitter**2
    r = d["w"] - predict_abscissa(theta, d)
    return -0.5 * jnp.sum(d["mask"] * (r**2 / var + jnp.log(var)))


def build_data(epoch, rvdata=None, m_star_prior=(1.0, 0.1), use_rvs=True, bounds=None):
    """Array-only pytree for `log_prob`: epoch astrometry + optional RVs + priors."""
    lo, hi = L.default_bounds()
    if bounds:
        for k, (a, b) in bounds.items():
            i = L.PARAM_NAMES.index(k); lo[i] = a; hi[i] = b
    from .data import RVData
    if rvdata is None and epoch.rv is not None:
        m = epoch.rv_mask > 0
        rvdata = RVData.from_arrays(epoch.rv_t_day[m] + epoch.ref_jd, epoch.rv[m], epoch.rv_err[m])
    if rvdata is None:
        rvdata = RVData.from_arrays([], [], [], n_pad=1); use_rvs = False
    rv = rvdata.as_jax()
    # RVData stores times relative to the DR3 epoch (JD 2457389.0); re-reference to this dataset's epoch
    rv["t"] = rv["t"] + (GAIA_REF_JD - epoch.ref_jd)
    return dict(epoch=epoch.as_jax(), rv=rv, rv_weight=jnp.asarray(1.0 if use_rvs else 0.0),
                m_star_mu=jnp.asarray(float(m_star_prior[0])), m_star_sigma=jnp.asarray(float(m_star_prior[1])),
                lo=jnp.asarray(lo), hi=jnp.asarray(hi))


def log_prob(theta, data):
    lp = L.log_prior(theta, data)
    ll = log_like_epoch(theta, data) + data["rv_weight"] * L.log_like_rv(theta, data)
    return jnp.where(jnp.isfinite(lp), lp + ll, -jnp.inf)
