"""Generate the mock-source bank used by the GPU benchmark notebooks (generate once, reuse everywhere).

    python scripts/make_mock_bank.py                      # data/mock_bank_10000.npz, seed 0
    python scripts/make_mock_bank.py --n 2000 --out data/mock_bank_2000.npz

Population (the same as the notebooks used to draw on the fly): solar-mass primary, dark companion with
M2 log-uniform in 0.2-10 Msun, distance log-uniform in 200-3000 pc, P log-uniform in 20-1500 d, e < 0.8,
isotropic orientation; scan patterns and errors from the 8 best-measured DR4 pre-release sources (median
per-CCD error < 0.5 mas); 15 epoch RVs per source with 1 km/s errors, drawn from the source's transits.
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import darkcomp  # noqa: E402,F401
from darkcomp import dr4, mock  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-rv", type=int, default=15)
ap.add_argument("--rv-err", type=float, default=1.0)
ap.add_argument("--out", default="data/mock_bank_10000.npz")
a = ap.parse_args()

t0 = time.perf_counter()
eps, _ = dr4.load_prerelease("data/dr4_prerelease/gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip")
templates = [e for e in eps.values() if np.median(e.err[e.mask > 0]) < 0.5]
rng = np.random.default_rng(a.seed)
truth = mock.physical_population(rng, a.n, m_star=1.0, m_2=np.exp(rng.uniform(np.log(0.2), np.log(10), a.n)),
                                 dist_pc=np.exp(rng.uniform(np.log(200), np.log(3000), a.n)), p_range=(20.0, 1500.0), e_max=0.8)
meta = dict(seed=a.seed, n=a.n, n_rv=a.n_rv, rv_err_kms=a.rv_err, m_star=1.0, m_2="log-uniform 0.2-10 Msun",
            dist_pc="log-uniform 200-3000", period_day="log-uniform 20-1500", e_max=0.8,
            templates="DR4 pre-release sources with median per-CCD error < 0.5 mas (8)", generator="scripts/make_mock_bank.py")
bank = mock.make_bank(rng, templates, truth, n_rv=a.n_rv, err_kms=a.rv_err, meta=meta)
mock.save_bank(a.out, bank)
print(f"{a.out}: {os.path.getsize(a.out)/1e6:.1f} MB, {mock.bank_info(a.out)}  ({time.perf_counter()-t0:.1f} s)")
