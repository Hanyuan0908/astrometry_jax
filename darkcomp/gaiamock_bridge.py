"""Mock epoch astrometry from gaiamock (El-Badry et al. 2024, arXiv:2411.00088).

gaiamock is the community tool for simulating Gaia epoch astrometry: it predicts the scan times,
scan angles and parallax factors for any sky position from the nominal scanning law (49,152 healpix
cells), the along-scan uncertainty as a function of G magnitude (per-CCD values from Holl et al.
2023, binned to one measurement per field-of-view transit), a random 10% transit rejection and,
for G < 13, an extra unmodelled noise term. This module calls gaiamock's own prediction functions
and wraps the result in `epoch.EpochAstrometry`, so the batch pipeline sees the same input as for
real data.

Setup (once): clone https://github.com/kareemelbadry/gaiamock, download and unzip healpix_scans.zip
into it, compile kepler_solve_astrometry.c against GSL, and point GAIAMOCK_DIR at the directory
(default ~/software/gaiamock). See docs/RUNNING.md.
"""
import os
import sys
import numpy as np

from . import epoch as E
from .constants import GAIA_REF_JD

GAIAMOCK_DIR = os.path.expanduser(os.environ.get("GAIAMOCK_DIR", "~/software/gaiamock"))
REF_JD = {"dr3": GAIA_REF_JD, "dr4": E.DR4_REF_JD, "dr5": 2458818.5}   # gaiamock.rescale_times_astrometry
_gm = None
_c_funcs = None


def available():
    return os.path.exists(os.path.join(GAIAMOCK_DIR, "gaiamock.py")) and \
        os.path.isdir(os.path.join(GAIAMOCK_DIR, "healpix_scans")) and \
        os.path.exists(os.path.join(GAIAMOCK_DIR, "kepler_solve_astrometry.so"))


def load():
    """Import gaiamock from GAIAMOCK_DIR (cached) and its compiled Kepler solver."""
    global _gm, _c_funcs
    if _gm is None:
        if not available():
            raise ImportError(f"gaiamock not found/complete in {GAIAMOCK_DIR}; set GAIAMOCK_DIR (see docs/RUNNING.md)")
        if GAIAMOCK_DIR not in sys.path:
            sys.path.insert(0, GAIAMOCK_DIR)
        import gaiamock  # noqa: E402
        _gm = gaiamock
        _c_funcs = gaiamock.read_in_C_functions()
    return _gm, _c_funcs


def random_sky(rng, n):
    """Isotropic (ra, dec) in degrees."""
    ra = rng.uniform(0.0, 360.0, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    return ra, dec


def g_from_distance(abs_mag_g, dist_pc, a_g=0.0):
    """Apparent G from absolute magnitude, distance and extinction."""
    return abs_mag_g + 5.0 * np.log10(np.asarray(dist_pc, float) / 10.0) + a_g


def per_ccd_error(g_mag):
    """gaiamock's per-CCD along-scan uncertainty [mas] as a function of G."""
    gm, _ = load()
    return gm.al_uncertainty_per_ccd_interp(np.asarray(g_mag, float))


def simulate(rng, truth, ra, dec, g_mag, data_release="dr4", n_pad=None):
    """List of EpochAstrometry, one per source, predicted by gaiamock.

    truth: dict as from `mock.random_population`/`mock.physical_population` (period [d], ecc, a0_mas,
    inc, Omega, omega [rad], t_peri [d rel. reference epoch], parallax [mas], pmra, pmdec [mas/yr]).
    ra, dec [deg] and g_mag: scalars or length-n arrays. Sources with d_ra_mas/d_dec_mas offsets
    in `truth` are shifted accordingly (gaiamock itself places the source at the reference position).
    Measurements are gaiamock's default: one per field-of-view transit (8 CCDs binned), with the
    G-dependent uncertainty, a random 10% of transits dropped and, for G < 13, extra noise.
    gaiamock draws from numpy's global RNG, which is seeded from `rng` per source for reproducibility.
    """
    gm, c = load()
    n = len(truth["period"])
    ra, dec, g_mag = (np.broadcast_to(np.asarray(x, float), (n,)) for x in (ra, dec, g_mag))
    ref_jd = REF_JD[data_release]
    out = []
    for i in range(n):
        p = {k: float(v[i]) for k, v in truth.items()}
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        t_yr, psi, plx_f, w, err = gm.predict_astrometry_binary_in_terms_of_a0(
            ra=ra[i], dec=dec[i], parallax=p["parallax"], pmra=p["pmra"], pmdec=p["pmdec"],
            period=p["period"], Tp=p["t_peri"], ecc=p["ecc"], omega=p["Omega"], inc=p["inc"], w=p["omega"],
            a0_mas=p["a0_mas"], phot_g_mean_mag=g_mag[i], data_release=data_release, c_funcs=c)
        w = w + p.get("d_ra_mas", 0.0) * np.sin(psi) + p.get("d_dec_mas", 0.0) * np.cos(psi)
        out.append(E.EpochAstrometry.from_arrays(ref_jd + t_yr * 365.25, w, err, plx_f, np.degrees(psi),
                                                 ref_jd=ref_jd, n_pad=n_pad))
    return out
