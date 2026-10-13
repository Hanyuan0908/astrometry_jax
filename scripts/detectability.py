"""Detection fraction of dark companions vs companion mass and distance, with the batch pipeline.

    DARKCOMP_PRECISION=32 python scripts/detectability.py --m2 0.3 0.6 1 1.4 2 3 5 10 --dist 200 500 1000 2000 --n 24

Mocks use the real scan patterns and per-CCD errors of the well-measured DR4 pre-release sources
(`mock.simulate`), a solar-mass primary by default, a dark companion of mass M2 at distance d, periods
log-uniform in --p-range, e uniform in [0, e_max], isotropic orientation. A source counts as detected when
the Thiele-Innes significance exceeds --sig. Writes a table, an .npz and a figure.
"""
import argparse, os, sys, time, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, jax, jax.numpy as jnp
import darkcomp
from darkcomp import dr4, batch, mock

ap = argparse.ArgumentParser()
ap.add_argument("--m2", type=float, nargs="+", default=[0.3, 0.6, 1.0, 1.4, 2.0, 3.0, 5.0, 10.0])
ap.add_argument("--dist", type=float, nargs="+", default=[200.0, 500.0, 1000.0, 2000.0])
ap.add_argument("--n", type=int, default=24, help="realisations per (M2, distance) cell")
ap.add_argument("--m-star", type=float, default=1.0)
ap.add_argument("--p-range", type=float, nargs=2, default=[30.0, 1500.0])
ap.add_argument("--e-max", type=float, default=0.6)
ap.add_argument("--sig", type=float, default=5.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mem-gb", type=float, default=1.0)
ap.add_argument("--noise-scale", type=float, default=1.0, help="multiply the template per-CCD errors (1 = the bright G~6-13 pre-release stars; ~5 mimics G~16-17)")
ap.add_argument("--all-templates", action="store_true", help="also use the G~19-20 templates with mas-level errors")
ap.add_argument("--out", default="docs/figures/detectability")
args = ap.parse_args()

root = os.path.join(os.path.dirname(__file__), "..")
eps, _ = dr4.load_prerelease(f"{root}/data/dr4_prerelease/gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip")
templates = [e for e in eps.values() if args.all_templates or np.median(e.err[e.mask > 0]) < 0.5]
errs = [np.median(e.err[e.mask > 0]) * args.noise_scale for e in templates]
print(f"precision {darkcomp.PRECISION}; {len(templates)} scan-pattern templates, median per-CCD error {min(errs):.2f}-{max(errs):.2f} mas (noise scale {args.noise_scale:g}); M* = {args.m_star} Msun")

rng = np.random.default_rng(args.seed)
M2, D = np.meshgrid(args.m2, args.dist, indexing="ij")                 # (n_m2, n_dist)
m2 = np.repeat(M2.ravel(), args.n); dist = np.repeat(D.ravel(), args.n)
truth = mock.physical_population(rng, len(m2), m_star=args.m_star, m_2=m2, dist_pc=dist,
                                 p_range=tuple(args.p_range), e_max=args.e_max)
dd = batch.stack_epochs(mock.simulate(rng, templates, truth, noise_scale=args.noise_scale))
periods = np.geomspace(20, 3000, 300)
n_src, n_obs = dd["w"].shape
sc, ch = batch.plan_chunks(n_obs, n_src, args.mem_gb, dtype_bytes=darkcomp.PRECISION // 8)
print(f"{n_src} mock sources, {n_obs} abscissae max; {sc} sources x {ch} trials per step")
t0 = time.perf_counter()
out = batch.run_pipeline(dd, jnp.asarray(periods), chunk=ch, source_chunk=sc, refine_chunk=min(max(4 * sc, 64), n_src),
                         m_star=args.m_star)
jax.block_until_ready(out["period"]); dt = time.perf_counter() - t0
res = {k: np.asarray(v) for k, v in out.items() if k != "periodogram"}
print(f"pipeline {dt:.1f} s ({n_src / dt:.1f} sources/s incl. compile)")

shape = (len(args.m2), len(args.dist), args.n)
sig = res["significance"].reshape(shape); det = sig > args.sig
relP = np.abs(res["period"] / truth["period"] - 1).reshape(shape)
a0 = truth["a0_mas"].reshape(shape)
frac = det.mean(axis=2); fracP = (det & (relP < 0.05)).mean(axis=2)
m2rec = res["m_2"].reshape(shape)

hdr = "M2 [Msun] | " + " | ".join(f"{d:6.0f} pc" for d in args.dist)
print(f"\nDetection fraction (significance > {args.sig}); in brackets: detected AND period within 5%; median a0 [mas]")
print(hdr); print("-" * len(hdr))
for i, m in enumerate(args.m2):
    cells = " | ".join(f"{frac[i, j]:4.2f} ({fracP[i, j]:4.2f}) {np.median(a0[i, j]):6.3f}" for j in range(len(args.dist)))
    print(f"{m:9.2f} | {cells}")
print("\nMedian recovered M2 for detected sources (true M2 rows, distance columns):")
for i, m in enumerate(args.m2):
    cells = " | ".join((f"{np.median(m2rec[i, j][det[i, j]]):6.2f}" if det[i, j].any() else "   -  ") for j in range(len(args.dist)))
    print(f"{m:9.2f} | {cells}")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
np.savez(args.out + ".npz", m2=args.m2, dist=args.dist, n=args.n, noise_scale=args.noise_scale, m_star=args.m_star, frac=frac, fracP=fracP, sig=sig, relP=relP, a0=a0,
         m2rec=m2rec, truth_period=truth["period"].reshape(shape), truth_ecc=truth["ecc"].reshape(shape), truth_inc=truth["inc"].reshape(shape))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for j, d in enumerate(args.dist):
    ax[0].plot(args.m2, frac[:, j], "o-", label=f"{d:.0f} pc")
ax[0].set_xscale("log"); ax[0].set_xlabel("companion mass M2 [Msun]"); ax[0].set_ylabel(f"detection fraction (> {args.sig:g} sigma)")
ax[0].set_ylim(-0.02, 1.02); ax[0].legend(title=f"M* = {args.m_star:g} Msun, P in {args.p_range[0]:g}-{args.p_range[1]:g} d, noise x{args.noise_scale:g}", fontsize=8, loc="lower right"); ax[0].grid(alpha=0.3)
sc_ = ax[1].scatter(truth["a0_mas"], res["significance"], c=np.log10(truth["period"]), s=8, cmap="viridis")
ax[1].axhline(args.sig, c="r", lw=0.8); ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("true a0 [mas]"); ax[1].set_ylabel("significance"); plt.colorbar(sc_, ax=ax[1], label="log10 P [d]")
plt.tight_layout(); plt.savefig(args.out + ".png", dpi=130)
print("saved", args.out + ".npz", args.out + ".png")
