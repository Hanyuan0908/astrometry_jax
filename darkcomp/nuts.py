"""NUTS sampling of the 14-parameter posterior with numpyro, using JAX gradients.

The sampler works on a reparameterised vector that is better conditioned than the raw
physical parameters (log period, sqrt(e) cos/sin omega, etc.); `to_physical` maps it back
to `likelihood.PARAM_NAMES` order so any `log_prob(theta, data)` from this package can be
used unchanged. Priors are flat in the physical parameters, as in El-Badry et al. 2024
(plus the Gaussian prior on m_star), implemented by adding the log-Jacobian of the
reparameterisation. Start NUTS at the global mode found by the grid search + refinement:
it will not hop between period aliases on its own.
"""
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, init_to_value

from . import likelihood as L

U_NAMES = ["log_period", "sqe_cos_w", "sqe_sin_w", "cos_inc", "Omega", "log_m_star", "log_m_2",
           "log_parallax", "d_ra_mas", "d_dec_mas", "pmra", "pmdec", "t_peri", "gamma"]


def to_physical(u):
    period = jnp.exp(u[0])
    ecc = u[1] ** 2 + u[2] ** 2
    omega = jnp.arctan2(u[2], u[1])
    inc = jnp.arccos(u[3])
    return jnp.stack([period, ecc, jnp.exp(u[5]), jnp.exp(u[6]), inc, u[4], omega, jnp.exp(u[7]),
                      u[8], u[9], u[10], u[11], u[12], u[13]])


def to_unconstrained(theta):
    theta = np.asarray(theta)
    p = dict(zip(L.PARAM_NAMES, theta))
    se = np.sqrt(p["ecc"])
    return np.array([np.log(p["period"]), se * np.cos(p["omega"]), se * np.sin(p["omega"]), np.cos(p["inc"]),
                     p["Omega"], np.log(p["m_star"]), np.log(p["m_2"]), np.log(p["parallax"]),
                     p["d_ra_mas"], p["d_dec_mas"], p["pmra"], p["pmdec"], p["t_peri"], p["gamma"]])


def log_jacobian(u):
    """log |d theta / d u| so that a flat prior in theta becomes the right density in u."""
    # period = e^u0 -> u0; (sqe_cos, sqe_sin) -> (e, omega): |J| = 2 ; inc = acos(u3): |di/du3| = 1/sqrt(1-u3^2)
    # masses, parallax: e^u
    return u[0] + jnp.log(2.0) - 0.5 * jnp.log1p(-u[3] ** 2) + u[5] + u[6] + u[7]


def make_model(log_prob, data):
    def model():
        u = numpyro.sample("u", dist.ImproperUniform(constraints.real, (), (14,)))
        theta = to_physical(u)
        numpyro.deterministic("theta", theta)
        numpyro.factor("logp", log_prob(theta, data) + log_jacobian(u))
    return model


def run_nuts(log_prob, data, theta0, key=None, num_warmup=1000, num_samples=1000, num_chains=1,
             dense_mass=True, target_accept=0.9, progress=False):
    """Returns (samples_theta (n, 14), mcmc object). theta0: physical start vector (e.g. MAP)."""
    key = jax.random.PRNGKey(0) if key is None else key
    u0 = jnp.asarray(to_unconstrained(theta0))
    kernel = NUTS(make_model(log_prob, data), dense_mass=dense_mass, target_accept_prob=target_accept,
                  init_strategy=init_to_value(values={"u": u0}))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method="sequential", progress_bar=progress)
    mcmc.run(key)
    theta = np.asarray(mcmc.get_samples()["theta"])
    return theta, mcmc


def summarize(theta):
    """Median and 16/84 percentiles of fit and derived parameters."""
    d = jax.vmap(L.derived)(jnp.asarray(theta))
    out = {}
    for k, v in d.items():
        v = np.asarray(v)
        if v.ndim == 1:
            lo, med, hi = np.percentile(v, [16, 50, 84]); out[k] = (med, hi - med, med - lo)
    for k in ("inc", "Omega", "omega"):
        v = np.degrees(np.asarray(d[k])); v = v % 360 if k != "inc" else v
        lo, med, hi = np.percentile(v, [16, 50, 84]); out[k + "_deg"] = (med, hi - med, med - lo)
    return out
