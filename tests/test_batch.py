"""Batched pipeline and DR4 loader. Small sizes: memory on the dev machine is limited."""
import numpy as np
import jax.numpy as jnp
import pytest
from darkcomp import dr4, epoch as E, batch, mock
from tests.conftest import DATA

PRE = f"{DATA}/dr4_prerelease/gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip"


@pytest.fixture(scope="module")
def prerelease():
    eps, df = dr4.load_prerelease(PRE)
    return eps, df


def test_prerelease_loads_12_sources(prerelease):
    eps, df = prerelease
    assert len(eps) == 12 and all(e.n > 400 for e in eps.values())
    # used_by_agis_al filtering keeps 55-85% of CCD measurements
    assert 0.5 < df.used_al.mean() < 0.9


def test_single_stars_and_quasars_are_null(prerelease):
    eps, _ = prerelease
    periods = np.geomspace(20, 3000, 150)
    for sid, (name, grp) in dr4.PRERELEASE_SOURCES.items():
        if grp == "orbit":
            continue
        d = eps[sid].as_jax()
        _, chi2, _ = E.fit_single_star(d)
        assert 0.8 < float(chi2) / eps[sid].n < 1.3, (sid, grp)
        ll, be, bp = E.period_search(d, periods, n_ecc=4, n_phase=6, chunk=128)
        k = int(np.argmax(np.asarray(ll)))
        r = E.refine_profile(d, periods[k], float(be[k]), float(bp[k]))
        assert r["significance"] < 6.0, (sid, grp, r["significance"])


def test_prerelease_bh3_and_hd114762_detected(prerelease):
    eps, _ = prerelease
    periods = np.geomspace(20, 6000, 300)
    for sid, p_true in ((4318465066420528000, 4253.0), (3937211745905473024, 83.9)):
        d = eps[sid].as_jax()
        ll, be, bp = E.period_search(d, periods, n_ecc=5, n_phase=8, chunk=128)
        k = int(np.argmax(np.asarray(ll)))
        r = E.refine_profile(d, periods[k], float(be[k]), float(bp[k]), fit_jitter=True)
        assert r["significance"] > 50 and abs(r["period"] / p_true - 1) < 0.03
        assert r["chi2"] / (r["n"] - 13) < 1.3


def test_batch_pipeline_recovers_mock_orbits(prerelease):
    eps, _ = prerelease
    rng = np.random.default_rng(1)
    truth = mock.random_population(rng, 8, p_range=(50, 800), a0_range=(0.5, 3.0))
    # precise templates only (median per-CCD error < 0.5 mas): the G~19-20 sources have mas-level errors
    good = [e for e in eps.values() if np.median(e.err[e.mask > 0]) < 0.5]
    assert len(good) >= 6
    dd = batch.stack_epochs(mock.simulate(rng, good, truth))
    r = batch.run_pipeline(dd, np.geomspace(20, 2000, 200), chunk=128)
    P = np.asarray(r["period"]); sig = np.asarray(r["significance"])
    assert np.all(sig > 5)
    assert np.mean(np.abs(P / truth["period"] - 1) < 0.05) >= 0.75
    assert np.all(np.abs(np.asarray(r["parallax"]) - truth["parallax"]) < 0.1)


def test_float32_search_matches_float64(prerelease):
    """Mixed precision (float32 periodogram, float64 refinement) must reproduce the float64 pipeline."""
    eps, _ = prerelease
    rng = np.random.default_rng(2)
    truth = mock.random_population(rng, 8, p_range=(50, 800), a0_range=(0.5, 3.0))
    good = [e for e in eps.values() if np.median(e.err[e.mask > 0]) < 0.5]
    dd = batch.stack_epochs(mock.simulate(rng, good, truth))
    periods = np.geomspace(20, 2000, 300)
    r64 = batch.run_pipeline(dd, periods, chunk=128)
    rm = batch.run_pipeline(dd, periods, chunk=128, search_dtype=jnp.float32)
    assert rm["period"].dtype == jnp.float64
    assert np.array_equal(np.asarray(r64["significance"]) > 5, np.asarray(rm["significance"]) > 5)
    assert np.all(np.abs(np.asarray(rm["period"]) / np.asarray(r64["period"]) - 1) < 1e-3)
    assert np.all(np.abs(np.asarray(rm["chi2"]) / np.asarray(r64["chi2"]) - 1) < 1e-3)
    # the float32 periodogram itself stays float32 and ranks the same peak
    ll32, _, _ = batch.batch_period_search(batch.as_dtype(dd, jnp.float32), periods, chunk=128)
    assert ll32.dtype == jnp.float32
    assert np.array_equal(np.argmax(np.asarray(ll32), 1), np.argmax(np.asarray(r64["periodogram"]), 1))
