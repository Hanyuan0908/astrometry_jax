import numpy as np
import jax
import jax.numpy as jnp
from darkcomp.sampler import run_ensemble, init_ball
from darkcomp import fit


def test_ensemble_recovers_correlated_gaussian():
    cov = jnp.array([[1.0, 0.8], [0.8, 2.0]])
    cinv = jnp.linalg.inv(cov)
    mu = jnp.array([1.0, -2.0])
    logp = lambda th, data: -0.5 * (th - mu) @ cinv @ (th - mu)
    key = jax.random.PRNGKey(3)
    w = init_ball(key, mu, jnp.ones(2) * 0.1, 32)
    chain, lps, acc = run_ensemble(key, w, None, logp, 4000)
    flat = np.asarray(chain[1000:]).reshape(-1, 2)
    assert np.allclose(flat.mean(0), np.asarray(mu), atol=0.1)
    assert np.allclose(np.cov(flat.T), np.asarray(cov), atol=0.25)
    assert 0.2 < float(acc.mean()) < 0.9


def test_short_end_to_end_fit_runs(j0553):
    sol, rvs, paper = j0553
    res = fit.fit_source(sol, rvs, paper["m_star_prior"], n_walkers=32, n_burn=200, n_steps=100)
    assert res.chain.shape == (100, 32, 14)
    s = res.summary()
    assert abs(s["m_2"][0] - paper["m_2"]) < 0.3
    assert np.all(np.isfinite(res.log_prob))
