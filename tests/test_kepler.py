import numpy as np
import jax
import jax.numpy as jnp
from darkcomp.kepler import eccentric_anomaly, true_anomaly


def test_kepler_residual_small_over_grid():
    M = jnp.linspace(-10, 10, 401)[:, None]
    e = jnp.array([0.0, 0.1, 0.5, 0.9, 0.99])[None, :]
    E = eccentric_anomaly(M, e)
    resid = E - e * jnp.sin(E) - jnp.mod(M, 2 * jnp.pi)
    assert float(jnp.max(jnp.abs(resid))) < 1e-10


def test_kepler_gradients_match_finite_difference():
    f = lambda M, e: eccentric_anomaly(M, e)
    M0, e0, h = 1.3, 0.7, 1e-6
    dM = float(jax.grad(f, 0)(M0, e0)); de = float(jax.grad(f, 1)(M0, e0))
    assert abs(dM - (float(f(M0 + h, e0)) - float(f(M0 - h, e0))) / (2 * h)) < 1e-6
    assert abs(de - (float(f(M0, e0 + h)) - float(f(M0, e0 - h))) / (2 * h)) < 1e-6


def test_true_anomaly_circular_equals_mean():
    M = jnp.linspace(0, 2 * jnp.pi, 50, endpoint=False)
    f = true_anomaly(M, 0.0)
    assert np.allclose(np.mod(np.asarray(f), 2 * np.pi), np.asarray(M), atol=1e-10)


def test_vmap_and_jit_compose():
    g = jax.jit(jax.vmap(eccentric_anomaly, in_axes=(0, None)))
    out = g(jnp.linspace(0, 6, 1000), 0.3)
    assert out.shape == (1000,) and bool(jnp.all(jnp.isfinite(out)))
