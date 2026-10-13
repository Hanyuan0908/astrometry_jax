"""Affine-invariant ensemble sampler (Goodman & Weare 2010 stretch move) in pure JAX.

This is the emcee algorithm the paper used, rewritten so the whole chain is one
`lax.scan` over jitted steps: all walkers are evaluated in a single vmapped call per
half-ensemble, so it maps onto a GPU, and the function is itself vmappable over
sources. Walkers are split into two halves updated alternately (the "red-blue"
scheme), which keeps the move detailed-balanced while vectorising.
"""
from functools import partial
import jax
import jax.numpy as jnp


def _stretch_half(key, active, other, logp_active, log_prob, data, a):
    """Propose + accept for the `active` half using `other` as complementary ensemble."""
    n_act, ndim = active.shape
    k_z, k_j, k_u = jax.random.split(key, 3)
    # z ~ g(z) propto 1/sqrt(z) on [1/a, a]
    u = jax.random.uniform(k_z, (n_act,))
    z = ((a - 1.0) * u + 1.0) ** 2 / a
    j = jax.random.randint(k_j, (n_act,), 0, other.shape[0])
    proposal = other[j] + z[:, None] * (active - other[j])
    logp_prop = jax.vmap(log_prob, in_axes=(0, None))(proposal, data)
    log_accept = (ndim - 1.0) * jnp.log(z) + logp_prop - logp_active
    accept = jnp.log(jax.random.uniform(k_u, (n_act,))) < log_accept
    new = jnp.where(accept[:, None], proposal, active)
    new_logp = jnp.where(accept, logp_prop, logp_active)
    return new, new_logp, accept


@partial(jax.jit, static_argnames=("log_prob", "n_steps", "a"))
def run_ensemble(key, walkers, data, log_prob, n_steps, a=2.0):
    """Run `n_steps` stretch-move iterations.

    walkers: (n_walkers, ndim) initial positions, n_walkers even and >= 2*ndim.
    Returns chain (n_steps, n_walkers, ndim), log_prob (n_steps, n_walkers) and
    acceptance fraction per walker.
    """
    n_walkers = walkers.shape[0]
    half = n_walkers // 2
    logp0 = jax.vmap(log_prob, in_axes=(0, None))(walkers, data)

    def step(carry, key):
        w, lp, n_acc = carry
        k1, k2 = jax.random.split(key)
        w0, lp0, acc0 = _stretch_half(k1, w[:half], w[half:], lp[:half], log_prob, data, a)
        w1, lp1, acc1 = _stretch_half(k2, w[half:], w0, lp[half:], log_prob, data, a)
        w = jnp.concatenate([w0, w1]); lp = jnp.concatenate([lp0, lp1])
        n_acc = n_acc + jnp.concatenate([acc0, acc1])
        return (w, lp, n_acc), (w, lp)

    keys = jax.random.split(key, n_steps)
    (w, lp, n_acc), (chain, lps) = jax.lax.scan(step, (walkers, logp0, jnp.zeros(n_walkers)), keys)
    return chain, lps, n_acc / n_steps


def init_ball(key, centre, scale, n_walkers):
    """Gaussian ball of walkers around `centre` with per-parameter `scale`."""
    return centre + scale * jax.random.normal(key, (n_walkers, centre.shape[0]))


def autocorr_time(x, c=5.0):
    """Integrated autocorrelation time per parameter (emcee's estimator) on an (n_steps, n_walkers, ndim) chain."""
    import numpy as np
    x = np.asarray(x)
    n, nw, nd = x.shape
    taus = []
    for d in range(nd):
        y = x[:, :, d] - x[:, :, d].mean(axis=0)
        f = np.zeros(n)
        for w in range(nw):
            fw = np.fft.ifft(np.abs(np.fft.fft(y[:, w], n=2 * n)) ** 2)[:n].real
            f += fw / fw[0]
        f /= nw
        tau = 2.0 * np.cumsum(f) - 1.0
        m = np.arange(n) < c * tau
        idx = np.argmin(m) if np.any(~m) else n - 1
        taus.append(tau[idx])
    return np.array(taus)
