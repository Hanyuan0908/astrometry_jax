"""Joint Gaia + RV log-likelihood (El-Badry et al. 2024, eqs. 13-16) in JAX.

Free parameters (14, in this order; `PARAM_NAMES`):
  period [d], ecc, m_star [Msun], m_2 [Msun], inc [rad], Omega [rad], omega [rad],
  parallax [mas], d_ra_mas, d_dec_mas, pmra [mas/yr], pmdec [mas/yr],
  t_peri [d after J2016.0], gamma [km/s]
Optionally a 15th parameter, the G-band flux ratio eps (Section 5.3).

All functions take a `data` pytree built by `build_data`, which has fixed array shapes
so `log_prob` can be jit-compiled once and vmapped over walkers (and over sources,
if the RV arrays are padded to a common length). The pytree holds only arrays: the
solution type is inferred from the length of `gaia_mean` (12 = Orbital, 15 =
AstroSpectroSB1), which is static under jit, and the RV term is switched on/off by
the scalar `rv_weight`.
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import orbit

PARAM_NAMES = ["period", "ecc", "m_star", "m_2", "inc", "Omega", "omega", "parallax",
               "d_ra_mas", "d_dec_mas", "pmra", "pmdec", "t_peri", "gamma"]
N_PARAMS = len(PARAM_NAMES)


def unpack(theta):
    d = {k: theta[..., i] for i, k in enumerate(PARAM_NAMES)}
    d["eps"] = theta[..., 14] if theta.shape[-1] > 14 else jnp.zeros_like(d["period"])
    return d


def derived(theta):
    """Physical quantities implied by a parameter vector."""
    p = unpack(theta)
    q = p["m_2"] / p["m_star"]
    a = orbit.semimajor_axis_au(p["period"], p["m_star"], p["m_2"])
    a1 = orbit.a1_au(a, q)
    a0 = orbit.photocentre_a0_mas(a, p["parallax"], q, p["eps"])
    K = orbit.rv_semi_amplitude_kms(a1, p["period"], p["ecc"], p["inc"])
    A, B, F, G = orbit.thiele_innes_ABFG(a0, p["omega"], p["Omega"], p["inc"])
    C, H = orbit.thiele_innes_CH(a1, p["omega"], p["inc"])
    return dict(p, q=q, a_au=a, a1_au=a1, a0_mas=a0, K=K, A=A, B=B, F=F, G=G, C=C, H=H,
                f_m_rv=orbit.rv_mass_function(p["period"], K, p["ecc"]),
                f_m_ast=orbit.astrometric_mass_function(a0, p["parallax"], p["period"]))


def gaia_vector(theta, is_sb1):
    """Predicted theta_Gaia in archive order (eqs. 11/12)."""
    d = derived(theta)
    common = [d["d_ra_mas"], d["d_dec_mas"], d["parallax"], d["pmra"], d["pmdec"], d["A"], d["B"], d["F"], d["G"]]
    tail = [d["ecc"], d["period"], d["t_peri"]]
    if is_sb1:
        return jnp.stack(common + [d["C"], d["H"], d["gamma"]] + tail, axis=-1)
    return jnp.stack(common + tail, axis=-1)


def predict_rv(theta, t):
    d = derived(theta)
    return orbit.radial_velocity_kms(t, d["period"], d["ecc"], d["t_peri"], d["omega"], d["K"], d["gamma"])


# ----------------------------------------------------------------------------- data pytree

def build_data(solution, rvdata=None, m_star_prior=(1.0, 0.1), use_rvs=True, bounds=None):
    """Pack a GaiaSolution, RVData and priors into a jit-friendly pytree.

    bounds: optional dict name -> (lo, hi) overriding the default flat-prior box.
    """
    cov = np.asarray(solution.cov)
    cov_inv = np.linalg.inv(cov)
    # whitening matrix W = L^{-1} with cov = L L^T, so chi2 = |W r|^2: a sum of squares, which stays
    # accurate in float32 where the quadratic form r^T cov^{-1} r can cancel catastrophically
    whiten = np.linalg.inv(np.linalg.cholesky(cov))
    lo, hi = default_bounds()
    if bounds:
        for k, (a, b) in bounds.items():
            i = PARAM_NAMES.index(k); lo[i] = a; hi[i] = b
    if rvdata is None:
        from .data import RVData
        rvdata = RVData.from_arrays([], [], [], n_pad=1)
        use_rvs = False
    return dict(
        gaia_mean=jnp.asarray(solution.mean),
        gaia_cov_inv=jnp.asarray(cov_inv),
        gaia_whiten=jnp.asarray(whiten),
        rv=rvdata.as_jax(),
        rv_weight=jnp.asarray(1.0 if use_rvs else 0.0),
        m_star_mu=jnp.asarray(float(m_star_prior[0])),
        m_star_sigma=jnp.asarray(float(m_star_prior[1])),
        lo=jnp.asarray(lo), hi=jnp.asarray(hi),
    )


def default_bounds():
    lo = np.full(N_PARAMS, -np.inf); hi = np.full(N_PARAMS, np.inf)
    box = {"period": (1.0, 1e5), "ecc": (0.0, 0.99), "m_star": (0.05, 20.0), "m_2": (0.0, 100.0),
           "inc": (0.0, np.pi), "parallax": (1e-3, 1e3)}
    for k, (a, b) in box.items():
        i = PARAM_NAMES.index(k); lo[i] = a; hi[i] = b
    return lo, hi


# ----------------------------------------------------------------------------- log-probabilities

def is_sb1(data):
    return data["gaia_mean"].shape[0] == 15


def log_like_gaia(theta, data):
    r = gaia_vector(theta, is_sb1(data)) - data["gaia_mean"]
    z = data["gaia_whiten"] @ r
    return -0.5 * jnp.sum(z**2)


def log_like_rv(theta, data):
    rv = data["rv"]
    resid = (predict_rv(theta, rv["t"]) - rv["rv"]) / rv["err"]
    return -0.5 * jnp.sum(rv["mask"] * resid**2)


def rv_chi2(theta, data):
    return -2.0 * log_like_rv(theta, data)


def log_prior(theta, data):
    inside = jnp.all((theta[:N_PARAMS] >= data["lo"]) & (theta[:N_PARAMS] <= data["hi"]))
    m_star = theta[2]
    lp = -0.5 * ((m_star - data["m_star_mu"]) / data["m_star_sigma"]) ** 2
    return jnp.where(inside, lp, -jnp.inf)


def residuals(theta, data):
    """Whitened residual vector r with log_prob = -|r|^2/2 inside the prior box (Gaia, RVs, m_star prior)."""
    z = data["gaia_whiten"] @ (gaia_vector(theta, is_sb1(data)) - data["gaia_mean"])
    rv = data["rv"]
    r_rv = jnp.sqrt(data["rv_weight"] * rv["mask"]) * (predict_rv(theta, rv["t"]) - rv["rv"]) / rv["err"]
    r_prior = (theta[2] - data["m_star_mu"]) / data["m_star_sigma"]
    return jnp.concatenate([z, r_rv, r_prior[None]])


def log_prob(theta, data):
    """Full log posterior for one parameter vector. jit/vmap/grad-able."""
    lp = log_prior(theta, data)
    ll = log_like_gaia(theta, data) + data["rv_weight"] * log_like_rv(theta, data)
    return jnp.where(jnp.isfinite(lp), lp + ll, -jnp.inf)


log_prob_batched = jax.vmap(log_prob, in_axes=(0, None))
