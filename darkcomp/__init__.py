"""darkcomp: joint Gaia astrometry + RV orbit fitting for dark companions, in JAX.

Implements the method of El-Badry et al. 2024 (arXiv:2405.00089), Section 5.
Everything numerical is float64; `darkcomp` enables x64 on import.
"""
import jax

jax.config.update("jax_enable_x64", True)

from . import constants, kepler, orbit, gaia, data, likelihood, sampler, fit  # noqa: E402,F401

__all__ = ["constants", "kepler", "orbit", "gaia", "data", "likelihood", "sampler", "fit"]
