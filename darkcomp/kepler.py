"""Differentiable, vectorised Kepler-equation solver.

E - e sin E = M is solved by Newton iteration with a fixed iteration count (so it
jit-compiles to straight-line code with no data-dependent control flow, which is
what makes it cheap under vmap on a GPU). Gradients are supplied analytically via
implicit differentiation rather than by unrolling the Newton loop.
"""
import jax
import jax.numpy as jnp

N_NEWTON = 12


def _solve(M, e):
    M = jnp.mod(M, 2.0 * jnp.pi)
    # Starter (Danby 1988): good for all e < 1, converges quadratically within a few steps
    E0 = M + 0.85 * e * jnp.sign(jnp.sin(M))
    # Fall back to E0 = M for e=0 (the sign term vanishes anyway for e->0)

    def body(_, E):
        f = E - e * jnp.sin(E) - M
        fp = 1.0 - e * jnp.cos(E)
        return E - f / fp

    return jax.lax.fori_loop(0, N_NEWTON, body, E0)


@jax.custom_jvp
def eccentric_anomaly(M, e):
    """Eccentric anomaly E for mean anomaly M (rad) and eccentricity e (0 <= e < 1).

    Broadcasts over any array shapes of M and e.
    """
    M, e = jnp.broadcast_arrays(jnp.asarray(M), jnp.asarray(e))
    return _solve(M, e)


@eccentric_anomaly.defjvp
def _eccentric_anomaly_jvp(primals, tangents):
    M, e = primals
    dM, de = tangents
    E = eccentric_anomaly(M, e)
    denom = 1.0 - e * jnp.cos(E)
    # From F(E, M, e) = E - e sin E - M = 0:
    dE = (dM + jnp.sin(E) * de) / denom
    return E, dE


def true_anomaly(M, e):
    """True anomaly f from mean anomaly M and eccentricity e."""
    E = eccentric_anomaly(M, e)
    return 2.0 * jnp.arctan2(jnp.sqrt(1.0 + e) * jnp.sin(E / 2.0), jnp.sqrt(1.0 - e) * jnp.cos(E / 2.0))


def mean_anomaly(t, period, t_peri):
    """Mean anomaly at times t for period and periastron time (same units as t)."""
    return 2.0 * jnp.pi * (t - t_peri) / period
