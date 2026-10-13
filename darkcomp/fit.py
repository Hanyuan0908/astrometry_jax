"""End-to-end fit of one source: initialise from the Gaia solution, resolve the +/- i
degeneracy by optimising both branches, then run the ensemble sampler on the winner.
"""
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import brentq, minimize

from . import orbit, likelihood as L
from .sampler import run_ensemble, init_ball


def m2_from_a0(a0_mas, parallax_mas, period_day, m_star, eps=0.0):
    """Solve a0 = a(P, M*+M2) * parallax * (q/(1+q) - eps/(1+eps)) for M2 (numpy, init only)."""
    def f(m2):
        a = float(orbit.semimajor_axis_au(period_day, m_star, m2))
        return float(orbit.photocentre_a0_mas(a, parallax_mas, m2 / m_star, eps)) - a0_mas
    return brentq(f, 1e-4, 1e3)


def initial_guesses(solution, rvdata=None, m_star=1.0):
    """Two Campbell-element initialisations (the +i / -i astrometric branches) from the Gaia solution.

    Returns a list of two (14,) numpy arrays in `likelihood.PARAM_NAMES` order.
    """
    s = solution
    A, B, F, G = (s["a_thiele_innes"], s["b_thiele_innes"], s["f_thiele_innes"], s["g_thiele_innes"])
    a0, omega, Omega, inc = orbit.campbell_from_thiele_innes(A, B, F, G)
    P, e, tp, plx = s["period"], s["eccentricity"], s["t_periastron"], s["parallax"]
    m2 = m2_from_a0(a0, plx, P, m_star)
    if s.is_sb1:
        gamma = s["center_of_mass_velocity"]
        # C = a1 sin w sin i, H = a1 cos w sin i fix the branch of omega
        omega_rv = np.arctan2(s["c_thiele_innes"], s["h_thiele_innes"])
        if np.cos(omega_rv - omega) < 0:
            omega += np.pi; Omega += np.pi
    elif rvdata is not None and rvdata.n > 0:
        gamma = float(np.average(rvdata.rv[rvdata.mask > 0], weights=1.0 / rvdata.err[rvdata.mask > 0] ** 2))
    else:
        gamma = 0.0
    base = np.array([P, e, m_star, m2, inc, Omega, omega, plx, 0.0, 0.0, s["pmra"], s["pmdec"], tp, gamma])
    flipped = base.copy(); flipped[5] += np.pi; flipped[6] += np.pi   # (w+pi, W+pi) == i -> -i
    return [base, flipped]


def _scale_for(solution, theta0):
    """Per-parameter jitter scale for the initial walker ball."""
    sc = np.zeros(L.N_PARAMS)
    sc[0] = solution.sigma("period") * 0.1
    sc[1] = solution.sigma("eccentricity") * 0.1
    sc[2] = 0.01; sc[3] = 0.01
    sc[4:7] = 0.01
    sc[7] = solution.sigma("parallax") * 0.1
    sc[8] = solution.sigma("d_ra_mas") * 0.1; sc[9] = solution.sigma("d_dec_mas") * 0.1
    sc[10] = solution.sigma("pmra") * 0.1; sc[11] = solution.sigma("pmdec") * 0.1
    sc[12] = solution.sigma("t_periastron") * 0.1
    sc[13] = 0.05
    return np.maximum(sc, 1e-6)


def optimise(theta0, data, maxiter=2000):
    """Gradient-based MAP from theta0 (L-BFGS on the JAX gradient)."""
    f = jax.jit(lambda th: -L.log_prob(th, data))
    g = jax.jit(jax.grad(lambda th: -L.log_prob(th, data)))
    theta0 = np.asarray(theta0, dtype=float)
    if not np.isfinite(float(f(jnp.asarray(theta0)))):
        return theta0, -np.inf
    res = minimize(lambda x: float(f(jnp.asarray(x))), theta0, jac=lambda x: np.asarray(g(jnp.asarray(x))),
                   method="L-BFGS-B", options={"maxiter": maxiter})
    return res.x, -float(res.fun)


@dataclass
class FitResult:
    chain: np.ndarray        # (n_steps, n_walkers, ndim) post burn-in
    log_prob: np.ndarray     # (n_steps, n_walkers)
    acceptance: np.ndarray
    map_theta: np.ndarray
    map_log_prob: float
    branch: int              # which of the two initialisations won

    @property
    def flat(self):
        return self.chain.reshape(-1, self.chain.shape[-1])

    def summary(self):
        """Posterior medians and 1-sigma intervals of fit + derived parameters."""
        flat = jnp.asarray(self.flat)
        d = jax.vmap(L.derived)(flat)
        d["inc_deg"] = jnp.degrees(d["inc"]); d["Omega_deg"] = jnp.degrees(jnp.mod(d["Omega"], 2 * jnp.pi))
        d["omega_deg"] = jnp.degrees(jnp.mod(d["omega"], 2 * jnp.pi))
        out = {}
        for k, v in d.items():
            v = np.asarray(v)
            if v.ndim == 1:
                lo, med, hi = np.percentile(v, [16, 50, 84])
                out[k] = (med, hi - med, med - lo)
        return out


def fit_source(solution, rvdata, m_star_prior, key=None, n_walkers=64, n_burn=3000, n_steps=3000,
               use_rvs=True, bounds=None, optimise_first=True):
    """Reproduce the paper's fit: 64 walkers, 3000 burn-in + 3000 production steps."""
    key = jax.random.PRNGKey(0) if key is None else key
    data = L.build_data(solution, rvdata, m_star_prior, use_rvs=use_rvs, bounds=bounds)
    guesses = initial_guesses(solution, rvdata, m_star=m_star_prior[0])
    best = None
    for b, th0 in enumerate(guesses):
        if optimise_first:
            th, lp = optimise(th0, data)
        else:
            th, lp = np.asarray(th0), float(L.log_prob(jnp.asarray(th0), data))
        if best is None or lp > best[1]:
            best = (th, lp, b)
    th_map, lp_map, branch = best
    scale = _scale_for(solution, th_map)
    k1, k2, k3 = jax.random.split(key, 3)
    walkers = init_ball(k1, jnp.asarray(th_map), jnp.asarray(scale), n_walkers)
    chain, lps, _ = run_ensemble(k2, walkers, data, L.log_prob, n_burn)
    chain, lps, acc = run_ensemble(k3, chain[-1], data, L.log_prob, n_steps)
    return FitResult(np.asarray(chain), np.asarray(lps), np.asarray(acc), np.asarray(th_map), float(lp_map), branch)
