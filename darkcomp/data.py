"""Radial-velocity data containers (padded, fixed-shape, so they vmap/jit cleanly)."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import jax.numpy as jnp

from .constants import GAIA_REF_JD


@dataclass
class RVData:
    t: np.ndarray        # days since Gaia reference epoch (JD 2457389.0)
    rv: np.ndarray       # km/s
    err: np.ndarray      # km/s
    mask: np.ndarray     # 1.0 for real points, 0.0 for padding

    @classmethod
    def from_arrays(cls, jd, rv, err, n_pad=None):
        jd, rv, err = (np.asarray(x, dtype=float) for x in (jd, rv, err))
        n = len(jd)
        n_pad = n if n_pad is None else max(n_pad, n)
        t = np.zeros(n_pad); r = np.zeros(n_pad); e = np.ones(n_pad); m = np.zeros(n_pad)
        t[:n] = jd - GAIA_REF_JD; r[:n] = rv; e[:n] = err; m[:n] = 1.0
        return cls(t, r, e, m)

    @property
    def n(self):
        return int(self.mask.sum())

    def as_jax(self):
        return {k: jnp.asarray(getattr(self, k)) for k in ("t", "rv", "err", "mask")}


def load_rv_csv(path, name=None, n_pad=None):
    """Load RVs from a CSV with columns name,hjd_utc,rv_kms,rv_err_kms[,instrument]."""
    df = pd.read_csv(path)
    if name is not None:
        df = df[df["name"] == name]
        if len(df) == 0:
            raise KeyError(f"no RVs for {name} in {path}")
    return RVData.from_arrays(df["hjd_utc"].values, df["rv_kms"].values, df["rv_err_kms"].values, n_pad=n_pad)
