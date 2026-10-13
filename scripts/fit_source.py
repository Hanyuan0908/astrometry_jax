"""Fit one source from the paper's sample using the bundled fixtures.

    python scripts/fit_source.py J0553-1349 [--steps 3000] [--burn 3000] [--no-rvs]

Uses data/gaia_dr3_nss_fixture.json if the source is there, else queries the archive.
"""
import argparse, time, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from darkcomp import gaia, data as D, fit, sampler  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--burn", type=int, default=3000)
ap.add_argument("--walkers", type=int, default=64)
ap.add_argument("--no-rvs", action="store_true")
args = ap.parse_args()

stars = pd.read_csv(f"{ROOT}/elbadry2024_star_params.csv", dtype={"source_id": str}).set_index("name")
sid = int(stars.loc[args.name, "source_id"])
m_prior = (float(stars.loc[args.name, "mstar"]), float(stars.loc[args.name, "mstar_err"]))
try:
    sol = gaia.load_solution(sid, fixture_path=f"{ROOT}/gaia_dr3_nss_fixture.json")
except KeyError:
    sol = gaia.load_solution(sid)
rvs = D.load_rv_csv(f"{ROOT}/elbadry2024_rvs.csv", name=args.name)
print(f"{args.name}: Gaia {sol.solution_type}, {rvs.n} RVs, M* prior {m_prior}")

t0 = time.time()
res = fit.fit_source(sol, rvs, m_prior, n_walkers=args.walkers, n_burn=args.burn, n_steps=args.steps,
                     use_rvs=not args.no_rvs)
print(f"done in {time.time()-t0:.1f}s; branch {res.branch}, MAP lnP={res.map_log_prob:.2f}, "
      f"mean acceptance {res.acceptance.mean():.2f}")
print("autocorr time (steps):", np.round(sampler.autocorr_time(res.chain), 1))
for k, (med, up, lo) in res.summary().items():
    if k in ("period", "ecc", "m_star", "m_2", "inc_deg", "Omega_deg", "omega_deg", "parallax", "t_peri",
             "gamma", "K", "a_au", "a0_mas", "f_m_rv", "f_m_ast"):
        print(f"{k:10s} = {med:12.5f} +{up:.5f} -{lo:.5f}")
