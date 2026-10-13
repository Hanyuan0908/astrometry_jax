"""Batched (many-source) epoch-astrometry pipeline: the GPU path.

All sources are padded to a common number of measurements and stacked along a leading axis;
every stage is then a `vmap`/`lax.map` over that axis with fixed shapes, so one jit compile
serves any number of sources and the work maps onto a GPU. Memory is controlled by `chunk`,
the number of (source, trial-orbit) pairs whose (n_obs, 9) design matrices exist at once.

Stages: single-star fit -> profile-likelihood period search -> gradient refinement (jaxopt
L-BFGS, vmapped) -> Campbell elements, significance and companion-mass estimate. An optional
vmapped ensemble-sampler posterior per source is in `batch_posterior`.
"""
import numpy as np
import jax
import jax.numpy as jnp
import jaxopt

from . import epoch as E, orbit
from .constants import GM_SUN_AU3_DAY2


# ----------------------------------------------------------------------------- stacking

def stack_epochs(epochs):
    """List of EpochAstrometry -> pytree with a leading source axis, padded to the longest."""
    n_max = max(len(e.t_day) for e in epochs)
    padded = [E.EpochAstrometry.from_arrays(e.t_day + e.ref_jd, e.w, e.err, e.plx_factor, np.degrees(e.psi),
                                            ref_jd=e.ref_jd, n_pad=n_max) if len(e.t_day) < n_max else e
              for e in epochs]
    # from_arrays pads with mask 0; existing padding rows (mask 0) are preserved by passing them through
    jx = [p.as_jax() for p in padded]
    return {k: jnp.stack([j[k] for j in jx]) for k in jx[0]}


def _take(dd, i):
    return jax.tree_util.tree_map(lambda x: x[i], dd)


# ----------------------------------------------------------------------------- stages

batch_single_star = jax.jit(jax.vmap(lambda d: E.fit_single_star(d)[:2]))


def batch_period_search(dd, periods, n_ecc=4, n_phase=8, chunk=256, e_max=0.6):
    """Per-source profile periodogram. Returns (loglike (S, nP), best_e (S, nP), best_phase (S, nP)).

    Outer `lax.map` over sources, inner chunked grid (no per-pair gathers); at most `chunk`
    (n_obs, 9) design matrices exist at a time.
    """
    periods = jnp.asarray(periods)
    eg = jnp.linspace(0.0, e_max, n_ecc); pg = jnp.linspace(0.0, 2 * jnp.pi, n_phase, endpoint=False)
    P, Ee, PH = jnp.meshgrid(periods, eg, pg, indexing="ij")
    P, Ee, PH = P.ravel(), Ee.ravel(), PH.ravel()

    def one_source(d):
        return E.profile_loglike_batch(P, Ee, PH, d, 0.0, chunk).reshape(len(periods), n_ecc * n_phase)

    ll = jax.lax.map(one_source, dd)
    k = jnp.argmax(ll, axis=-1)
    best = jnp.take_along_axis(ll, k[..., None], -1)[..., 0]
    Eg = Ee.reshape(len(periods), -1); PHg = PH.reshape(len(periods), -1)
    j = jnp.arange(len(periods))[None, :]
    return best, Eg[j, k], PHg[j, k]


def _refine_one(d, x0, maxiter, p_bounds):
    """Bounded L-BFGS over (log P, e, phase) on the profile likelihood."""
    lo = jnp.array([jnp.log(p_bounds[0]), 0.0, -10.0]); hi = jnp.array([jnp.log(p_bounds[1]), 0.95, 10.0 + 2 * jnp.pi])

    def loss(x):
        return -E.profile_loglike(jnp.exp(x[0]), x[1], x[2], d, 0.0)
    solver = jaxopt.LBFGSB(fun=loss, maxiter=maxiter, tol=1e-9)
    x, state = solver.run(jnp.array([jnp.log(x0[0]), x0[1], x0[2]]), bounds=(lo, hi))
    Pb, e, ph = jnp.exp(x[0]), x[1], x[2]
    beta, chi2, cov = E.weighted_lstsq(E.orbital_design(d, Pb, e, ph), d)
    A, B, F, G = beta[7], beta[5], beta[8], beta[6]
    a0, omega, Omega, inc = orbit.campbell_from_thiele_innes(A, B, F, G)
    ti = beta[5:]; cti = cov[5:, 5:]
    sig = jnp.sqrt(ti @ jnp.linalg.solve(cti, ti))
    return dict(period=Pb, ecc=e, phase=ph, chi2=chi2, a0_mas=a0, omega=omega, Omega=Omega, inc=inc,
                parallax=beta[2], pmra=beta[3], pmdec=beta[4], d_ra_mas=beta[0], d_dec_mas=beta[1],
                significance=sig, A=A, B=B, F=F, G=G)


def batch_refine(dd, period0, ecc0, phase0, maxiter=100, p_bounds=(1.0, 1e5), chunk=64):
    """Vmapped bounded refinement, `chunk` sources at a time (each holds a few design matrices)."""
    x0 = jnp.stack([jnp.asarray(period0), jnp.asarray(ecc0), jnp.asarray(phase0)], axis=-1)
    S = x0.shape[0]; n_pad = (-S) % chunk
    padf = lambda x: jnp.concatenate([x, jnp.repeat(x[:1], n_pad, axis=0)], axis=0)  # noqa: E731
    ddp = jax.tree_util.tree_map(padf, dd); x0p = padf(x0)
    ddp = jax.tree_util.tree_map(lambda x: x.reshape((-1, chunk) + x.shape[1:]), ddp)
    x0p = x0p.reshape(-1, chunk, 3)
    f = jax.vmap(lambda d, x: _refine_one(d, x, maxiter, p_bounds))
    out = jax.lax.map(lambda a: f(*a), (ddp, x0p))
    return jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:])[:S], out)


def mass_function_astrometric(a0_mas, parallax_mas, period_day):
    return (a0_mas / parallax_mas) ** 3 / (period_day / 365.25) ** 2


def m2_from_mass_function(fm, m_star, n_iter=30):
    """Solve fm = M2^3/(M*+M2)^2 for M2 by fixed-point iteration (vectorised)."""
    m2 = jnp.maximum(fm, 1e-3)
    for _ in range(n_iter):
        m2 = (fm * (m_star + m2) ** 2) ** (1.0 / 3.0)
    return m2


def top_peaks(ll, n_top):
    """Indices (S, n_top) of the n_top highest *local maxima* of each periodogram row (padded with the global max)."""
    left = jnp.concatenate([jnp.full(ll[:, :1].shape, -jnp.inf), ll[:, :-1]], axis=1)
    right = jnp.concatenate([ll[:, 1:], jnp.full(ll[:, :1].shape, -jnp.inf)], axis=1)
    is_peak = (ll >= left) & (ll >= right)
    masked = jnp.where(is_peak, ll, -jnp.inf)
    return jnp.argsort(-masked, axis=1)[:, :n_top]


def run_pipeline(dd, periods, n_ecc=4, n_phase=8, chunk=256, m_star=1.0, maxiter=40, refine_chunk=64, n_top=3,
                 e_max=0.6):
    """Whole search for a stacked batch. Returns a dict of per-source arrays (all jnp).

    The `n_top` highest periodogram peaks of every source are refined and the one with the
    lowest chi2 is kept, which guards against aliases and spurious high-e grid solutions.
    """
    periods = jnp.asarray(periods)
    S = dd["w"].shape[0]
    beta5, chi2_5 = batch_single_star(dd)
    ll, be, bp = batch_period_search(dd, periods, n_ecc, n_phase, chunk, e_max)
    ks = top_peaks(ll, n_top)                      # (S, n_top)
    idx = jnp.arange(S)[:, None]
    rep = jax.tree_util.tree_map(lambda x: jnp.repeat(x, n_top, axis=0), dd)
    r = batch_refine(rep, periods[ks].ravel(), be[idx, ks].ravel(), bp[idx, ks].ravel(), maxiter,
                     p_bounds=(float(periods[0]) * 0.5, float(periods[-1]) * 2.0), chunk=refine_chunk)
    r = jax.tree_util.tree_map(lambda x: x.reshape((S, n_top) + x.shape[1:]), r)
    best = jnp.argmin(r["chi2"], axis=1)
    r = jax.tree_util.tree_map(lambda x: x[jnp.arange(S), best], r)
    n = dd["mask"].sum(axis=1)
    r["chi2_single"] = chi2_5; r["n_obs"] = n; r["dchi2"] = chi2_5 - r["chi2"]
    r["parallax_single"] = beta5[:, 2]
    r["f_m_ast"] = mass_function_astrometric(r["a0_mas"], r["parallax"], r["period"])
    r["m_2"] = m2_from_mass_function(r["f_m_ast"], m_star)
    r["periodogram"] = ll
    return r
