"""Fixed-iteration Levenberg-Marquardt for least-squares objectives, jit/vmap-safe in any precision.

Used for the epoch-astrometry refinement (3 nonlinear parameters) and the DR3 MAP fit (14
parameters in the unconstrained parameterisation). Unlike line-search quasi-Newton methods it does
not rely on resolving tiny function differences, so it behaves identically in float32 and float64.
"""
import jax
import jax.numpy as jnp


def levenberg_marquardt(resid_fn, x0, n_iter=30, lo=None, hi=None, lam0=1e-2, lam_up=10.0, lam_down=3.0):
    """Minimise |resid_fn(x)|^2. Returns (x, chi2). Bounds are enforced by clipping the step.

    Each iteration: J = d resid / dx (forward mode), solve (J^T J + lam diag(J^T J)) step = -J^T r,
    accept if chi2 decreases (lam /= lam_down), else reject and lam *= lam_up.
    """
    jac = jax.jacfwd(resid_fn)
    x0 = jnp.asarray(x0)
    lo = jnp.full_like(x0, -jnp.inf) if lo is None else jnp.asarray(lo, x0.dtype)
    hi = jnp.full_like(x0, jnp.inf) if hi is None else jnp.asarray(hi, x0.dtype)
    x0 = jnp.clip(x0, lo, hi)

    def body(_, carry):
        x, chi2, lam = carry
        J = jac(x); r = resid_fn(x)
        H = J.T @ J; g = J.T @ r
        D = jnp.diag(jnp.diag(H)) + 1e-12 * jnp.eye(H.shape[0], dtype=H.dtype)
        step = -jnp.linalg.solve(H + lam * D, g)
        xn = jnp.clip(x + step, lo, hi)
        chi2n = jnp.sum(resid_fn(xn) ** 2)
        accept = chi2n < chi2
        x = jnp.where(accept, xn, x); chi2 = jnp.where(accept, chi2n, chi2)
        lam = jnp.where(accept, lam / lam_down, lam * lam_up)
        return x, chi2, lam

    chi2_0 = jnp.sum(resid_fn(x0) ** 2)
    x, chi2, _ = jax.lax.fori_loop(0, n_iter, body, (x0, chi2_0, jnp.asarray(lam0, x0.dtype)))
    return x, chi2
