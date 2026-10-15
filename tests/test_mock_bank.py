"""The mock bank (generate once, reuse) must reproduce exactly what stack_epochs gives for the same draws."""
import os, sys
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import darkcomp  # noqa: E402,F401
from darkcomp import batch, dr4, mock  # noqa: E402

ZIP = os.path.join(os.path.dirname(__file__), "..", "data", "dr4_prerelease", "gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip")
BANK = os.path.join(os.path.dirname(__file__), "..", "data", "mock_bank_10000.npz")


def _population(seed, n):
    rng = np.random.default_rng(seed)
    truth = mock.physical_population(rng, n, m_star=1.0, m_2=np.exp(rng.uniform(np.log(0.2), np.log(10), n)),
                                     dist_pc=np.exp(rng.uniform(np.log(200), np.log(3000), n)), p_range=(20.0, 1500.0), e_max=0.8)
    return rng, truth


def test_bank_roundtrip_matches_stack_epochs(tmp_path):
    eps, _ = dr4.load_prerelease(ZIP)
    templates = [e for e in eps.values() if np.median(e.err[e.mask > 0]) < 0.5]
    n = 10
    rng, truth = _population(5, n)
    sims = mock.add_rvs(rng, mock.simulate(rng, templates, truth), truth, n_rv=15, err_kms=1.0)
    ref = batch.stack_epochs(sims)
    rng, truth2 = _population(5, n)
    bank = mock.make_bank(rng, templates, truth2, n_rv=15, err_kms=1.0, meta=dict(test=True))
    path = tmp_path / "bank.npz"; mock.save_bank(path, bank)
    dd, tr = mock.load_bank(path, n, with_rvs=True)
    assert set(dd) == set(ref)
    for k in ref:
        a, b = np.asarray(ref[k]), np.asarray(dd[k])
        assert a.shape == b.shape, k
        tol = 1e-6 * np.abs(a).max() + 1e-7 if k == "w" else 1e-9      # w is stored in float32
        assert np.abs(a - b).max() < tol, (k, np.abs(a - b).max())
    for k in truth:
        assert np.allclose(tr[k], truth[k])
    dd_ast, _ = mock.load_bank(path, 4, with_rvs=False, offset=3)
    assert "rv" not in dd_ast and dd_ast["w"].shape[0] == 4
    assert np.allclose(np.asarray(dd_ast["w"]), np.asarray(dd["w"])[3:7])
    assert mock.bank_info(path)["n_rv"] == 15
    # the two views run through the pipeline to the same astrometric solution
    out = batch.run_pipeline(dd, jnp.asarray(np.geomspace(20, 3000, 300)), chunk=128, source_chunk=2, refine_chunk=8, n_trials=1200)
    assert out["period"].shape == (n,)


def test_committed_bank_is_consistent():
    info = mock.bank_info(BANK)
    assert info["n_sources"] == 10000 and info["n_rv"] == 15 and info["n_templates"] == 8
    dd, truth = mock.load_bank(BANK, 16, with_rvs=True)
    assert dd["w"].shape == (16, info["n_obs_max"]) and dd["rv"].shape == (16, 15)
    m = np.asarray(dd["mask"]) > 0
    assert 400 <= m.sum(axis=1).min() and m.sum(axis=1).max() <= info["n_obs_max"]
    assert np.all(np.asarray(dd["w"])[~m] == 0)
    assert len(truth["period"]) == 16 and truth["period"].min() >= 20 and truth["period"].max() <= 1500
