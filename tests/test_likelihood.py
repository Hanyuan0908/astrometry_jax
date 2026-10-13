"""Checks against the published J0553-1349 fit (El-Badry et al. 2024, Appendix C)."""
import numpy as np
import jax
import jax.numpy as jnp
from darkcomp import likelihood as L, fit


def paper_theta(paper):
    return jnp.array([paper["period"], paper["ecc"], paper["m_star"], paper["m_2"], np.radians(paper["inc_deg"]),
                      np.radians(paper["Omega_deg"]), np.radians(paper["omega_deg"]), paper["parallax"], 0.0, 0.0,
                      3.2349, 6.8568, paper["t_peri"], paper["gamma"]])


def test_derived_quantities_match_paper(j0553):
    sol, rvs, paper = j0553
    d = L.derived(paper_theta(paper))
    assert abs(float(d["a_au"]) - paper["a_au"]) < 0.005
    assert abs(float(d["a0_mas"]) - paper["a0_mas"]) < 0.01
    assert abs(float(d["K"]) - paper["K"]) < 0.1
    assert abs(float(d["f_m_rv"]) - paper["f_m_rv"]) < 0.002


def test_paper_solution_is_consistent_with_gaia_vector(j0553):
    sol, rvs, paper = j0553
    pred = np.asarray(L.gaia_vector(paper_theta(paper), sol.is_sb1))
    sig = np.sqrt(np.diag(sol.cov))
    pulls = (pred - sol.mean) / sig
    # A,B,F,G: paper found these all within ~2 sigma for this source
    for name in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"):
        assert abs(pulls[sol.names.index(name)]) < 3.0, (name, pulls)


def test_rv_sign_convention_gives_paper_chi2(j0553):
    """With the wrong omega / RV sign convention chi2/N would be in the hundreds."""
    sol, rvs, paper = j0553
    data = L.build_data(sol, rvs, paper["m_star_prior"])
    chi2 = float(L.rv_chi2(paper_theta(paper), data)) / rvs.n
    assert chi2 < 5.0, chi2
    flipped = paper_theta(paper).at[6].add(np.pi)
    assert float(L.rv_chi2(flipped, data)) / rvs.n > 50


def test_map_optimisation_recovers_paper_values(j0553):
    sol, rvs, paper = j0553
    data = L.build_data(sol, rvs, paper["m_star_prior"])
    best = max((fit.optimise(th0, data) for th0 in fit.initial_guesses(sol, rvs, paper["m_star"])), key=lambda r: r[1])
    th, lp = best
    d = {k: float(v) for k, v in L.derived(jnp.asarray(th)).items()}
    assert abs(d["period"] - paper["period"]) < 0.2
    assert abs(d["ecc"] - paper["ecc"]) < 0.005
    assert abs(d["K"] - paper["K"]) < 0.15
    assert abs(d["m_2"] - paper["m_2"]) < 0.15
    assert abs(np.degrees(d["inc"]) - paper["inc_deg"]) < 3
    assert abs(d["gamma"] - paper["gamma"]) < 0.1
    assert float(L.rv_chi2(jnp.asarray(th), data)) / rvs.n < 2.5


def test_log_prob_is_differentiable_and_batched(j0553):
    sol, rvs, paper = j0553
    data = L.build_data(sol, rvs, paper["m_star_prior"])
    th = paper_theta(paper)
    g = jax.grad(L.log_prob)(th, data)
    assert np.all(np.isfinite(np.asarray(g)))
    batch = jnp.stack([th, th * 1.0001])
    assert L.log_prob_batched(batch, data).shape == (2,)
