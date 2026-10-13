"""Run the whole pipeline at the precision given by DARKCOMP_PRECISION and dump results, so two
runs (32 and 64) can be compared:   DARKCOMP_PRECISION=32 python scripts/precision_check.py out32.npz
"""
import os, sys, time, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, jax, jax.numpy as jnp
import darkcomp
from darkcomp import dr4, batch, mock, epoch as E

print("precision", darkcomp.PRECISION, "dtype", jnp.ones(1).dtype)
eps, _ = dr4.load_prerelease("data/dr4_prerelease/gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip")
good = [e for e in eps.values() if np.median(e.err[e.mask > 0]) < 0.5]
rng = np.random.default_rng(0); truth = mock.random_population(rng, 64)
dd = batch.stack_epochs(mock.simulate(rng, good, truth)); periods = np.geomspace(20, 3000, 300)
r = batch.run_pipeline(dd, periods, chunk=256); jax.block_until_ready(r["period"])
t0 = time.perf_counter(); r = batch.run_pipeline(dd, periods, chunk=256); jax.block_until_ready(r["period"]); dt = time.perf_counter() - t0
out = {k: np.asarray(v) for k, v in r.items()}; out["truth_period"] = truth["period"]; out["truth_a0"] = truth["a0_mas"]; out["time"] = dt
# real sources, single-source path
ep = E.EpochAstrometry.from_bh3_table("data/gaia_bh3/epochast.dat"); d = ep.as_jax(); pg = np.geomspace(50, 8000, 600)
ll, be, bp = E.period_search(d, pg, chunk=128); k = int(np.argmax(np.asarray(ll))); rb = E.refine_profile(d, pg[k], float(be[k]), float(bp[k]))
out["bh3"] = np.array([rb["period"], rb["ecc"], rb["a0_mas"], np.degrees(rb["inc"]), rb["linear"]["parallax"], rb["linear"]["pmra"], rb["linear"]["pmdec"], rb["significance"], rb["chi2"]])
d = eps[3937211745905473024].as_jax(); pg = np.geomspace(20, 5000, 500)
ll, be, bp = E.period_search(d, pg, chunk=128); k = int(np.argmax(np.asarray(ll))); rh = E.refine_profile(d, pg[k], float(be[k]), float(bp[k]), fit_jitter=True)
out["hd114762"] = np.array([rh["period"], rh["ecc"], rh["a0_mas"], np.degrees(rh["inc"]), rh["linear"]["parallax"], rh["linear"]["pmra"], rh["linear"]["pmdec"], rh["significance"], rh["chi2"]])
np.savez(sys.argv[1], **out)
print("batch of 64: %.1fs; BH3 P %.1f a0 %.3f plx %.4f; HD114762 P %.2f a0 %.3f plx %.4f" % (dt, rb["period"], rb["a0_mas"], rb["linear"]["parallax"], rh["period"], rh["a0_mas"], rh["linear"]["parallax"]))
