"""darkcomp: joint Gaia astrometry + RV orbit fitting for dark companions, in JAX.

Implements the method of El-Badry et al. 2024 (arXiv:2405.00089), Section 5, and the
epoch-astrometry (DR4) search.

Precision is chosen once, at import, from the environment variable DARKCOMP_PRECISION
("64", the default, or "32"); the whole pipeline then runs in that dtype. It must be set
before `import darkcomp` (JAX reads the x64 flag on first use).
"""
import os
import jax

PRECISION = int(os.environ.get("DARKCOMP_PRECISION", "64"))
if PRECISION not in (32, 64):
    raise ValueError("DARKCOMP_PRECISION must be 32 or 64")
jax.config.update("jax_enable_x64", PRECISION == 64)

from . import constants, kepler, orbit, gaia, data, likelihood, sampler, fit  # noqa: E402,F401

__all__ = ["constants", "kepler", "orbit", "gaia", "data", "likelihood", "sampler", "fit"]
