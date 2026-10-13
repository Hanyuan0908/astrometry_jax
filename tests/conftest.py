import os, sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")


@pytest.fixture(scope="session")
def j0553():
    """Gaia solution, RVs and paper posterior medians for J0553-1349 (Table in Appendix C)."""
    from darkcomp import gaia, data
    sol = gaia.load_solution(2995961897685517312, fixture_path=f"{DATA}/gaia_dr3_nss_fixture.json")
    rvs = data.load_rv_csv(f"{DATA}/elbadry2024_rvs.csv", name="J0553-1349")
    paper = dict(period=189.10, a_au=0.851, a0_mas=1.229, ecc=0.3879, inc_deg=53.1, t_peri=-67.3,
                 Omega_deg=264.8, omega_deg=162.74, m_star=0.98, m_2=1.33, K=24.506, f_m_rv=0.2258,
                 gamma=41.033, parallax=2.505, chi2_per_rv=1.2, m_star_prior=(0.98, 0.06))
    return sol, rvs, paper
