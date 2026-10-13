"""Gaia BH3 epoch-astrometry benchmark (Gaia Collaboration, Panuzzo et al. 2024, Table 2, astrometry-only).

Memory note: period_search is chunked; keep grids small here (this test uses 28,800 points, ~11 MB/chunk).
"""
import numpy as np
import pytest
from darkcomp import epoch as E
from darkcomp.fit import m2_from_a0
from tests.conftest import DATA

PAPER = dict(period=(4194.7, 112.3), ecc=(0.7262, 0.0056), a0=(27.07, 0.56), inc=(110.659, 0.107),
             Omega=(136.200, 0.147), omega=(77.77, 0.66), tp_jd=(2458177.28, 0.98), plx=(1.6747, 0.0094),
             pmra=(-28.372, 0.077), pmdec=(-155.150, 0.129), fm=(32.03, 0.64))


@pytest.fixture(scope="module")
def bh3():
    ep = E.EpochAstrometry.from_bh3_table(f"{DATA}/gaia_bh3/epochast.dat")
    return ep, ep.as_jax()


def test_single_star_fit_is_terrible_for_bh3(bh3):
    ep, d = bh3
    _, chi2, _ = E.fit_single_star(d)
    assert float(chi2) / ep.n > 1000


def test_period_search_and_refine_recover_paper_orbit(bh3):
    ep, d = bh3
    periods = np.geomspace(50, 8000, 600)
    ll, be, bp = E.period_search(d, periods, n_ecc=6, n_phase=8, chunk=256)
    k = int(np.argmax(np.asarray(ll)))
    r = E.refine_profile(d, periods[k], float(be[k]), float(bp[k]))
    lin = r["linear"]
    got = dict(period=r["period"], ecc=r["ecc"], a0=r["a0_mas"], inc=np.degrees(r["inc"]),
               Omega=np.degrees(r["Omega"]) % 360, omega=np.degrees(r["omega"]) % 360,
               tp_jd=E.DR4_REF_JD + r["t_peri"], plx=lin["parallax"], pmra=lin["pmra"], pmdec=lin["pmdec"],
               fm=r["a0_mas"]**3 / lin["parallax"]**3 / (r["period"] / 365.25)**2)
    for key, (val, sig) in PAPER.items():
        assert abs(got[key] - val) < 2.0 * sig + 1e-3, (key, got[key], val, sig)
    assert r["chi2"] / (r["n"] - 12) < 1.2
    m2 = m2_from_a0(r["a0_mas"], lin["parallax"], r["period"], 0.76)
    assert abs(m2 - 32.70) < 1.5
