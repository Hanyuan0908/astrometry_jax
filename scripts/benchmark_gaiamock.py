"""Benchmark gaiamock against the real Gaia epoch astrometry we have (DR4 pre-release, 12 sources).

    python scripts/benchmark_gaiamock.py

A. Scanning law: gaiamock's predicted transits at each source's position vs the real transits.
   'same' = within 30 min (identical transit); 'visit' = within 12 h (same scan visit: gaiamock's 0.9 deg
   healpix cell can be swept one revolution or one field of view before/after the source, i.e. offsets of
   106, 253 or 360 min). Scan angles and parallax factors are compared on the visit matches.
B. Error model: gaiamock's per-CCD along-scan error at the source's G vs the real median per-CCD error.
C. Single-star residuals: for the 9 non-orbit sources, chi2/N of the 5-parameter fit on the real data vs
   on gaiamock realisations (does gaiamock's noise match the real errors?).
D. Gaia BH3 end to end: fit the real data, inject the fitted orbit into gaiamock realisations at BH3's
   position and G, refit, and compare recovered parameters, formal uncertainties and significance.
"""
import os, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, jax.numpy as jnp
import darkcomp
from darkcomp import dr4, epoch as E, mock, gaiamock_bridge as GB

gm, c_funcs = GB.load()
root = os.path.join(os.path.dirname(__file__), "..")
eps, df = dr4.load_prerelease(f"{root}/data/dr4_prerelease/gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip")
used = df[df.used_al & ~df.multipeak]
rng = np.random.default_rng(0)

print("A. Scanning law (DR4): real transits vs gaiamock prediction at the same position")
print(f"{'source':16s} {'G':>5s} {'N_real':>6s} {'N_gm':>5s} {'same':>5s} {'visit':>5s} {'dpsi[deg]':>9s} {'dplxf':>7s}   {'B. err/CCD real':>15s} {'gaiamock':>8s} {'excess':>6s}")
rows = []
for sid, e in eps.items():
    s = used[used.source_id == sid]; name = dr4.PRERELEASE_SOURCES[sid][0] or dr4.PRERELEASE_SOURCES[sid][1]
    tr = s.groupby("transit_id").agg(jd=("jd_tcb", "mean"), psi=("psi_deg", "mean"), plxf=("plx_factor", "first"))
    ra0, dec0, G = float(s.ra0.iloc[0]), float(s.dec0.iloc[0]), float(s.g_mag.iloc[0])
    tab = gm.get_gost_one_position(ra0, dec0, "dr4")
    psi_g, plxf_g, jd_g = gm.fetch_table_element(["scanAngle[rad]", "parallaxFactorAlongScan", "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"], tab)
    psi_g = np.degrees(np.asarray(psi_g)); plxf_g = np.asarray(plxf_g); jd_g = np.asarray(jd_g)
    i_near = np.abs(tr.jd.values[:, None] - jd_g[None, :]).argmin(axis=1); dt_min = (tr.jd.values - jd_g[i_near]) * 1440
    same = np.abs(dt_min) < 30; visit = np.abs(dt_min) < 720
    dpsi = (tr.psi.values[visit] - psi_g[i_near][visit] + 180) % 360 - 180
    dplxf = tr.plxf.values[visit] - plxf_g[i_near][visit]
    err_real = float(s.err.median()); err_gm = float(gm.al_uncertainty_per_ccd_interp(G)); excess = float(s.excess_noise.iloc[0])
    print(f"{name:16s} {G:5.2f} {len(tr):6d} {len(jd_g):5d} {same.sum():5d} {visit.sum():5d} {np.sqrt(np.mean(dpsi**2)):9.3f} {np.sqrt(np.mean(dplxf**2)):7.4f}   {err_real:15.3f} {err_gm:8.3f} {excess:6.2f}")
    rows.append(dict(sid=sid, name=name, G=G, ra0=ra0, dec0=dec0, n_real=len(tr), n_gm=len(jd_g), n_same=int(same.sum()), n_visit=int(visit.sum()), err_real=err_real, err_gm=err_gm, excess=excess))
rows = pd.DataFrame(rows)

print("\nC. Single-star fit chi2/N: real data vs gaiamock single-star realisations at the same G (median of 10)")
print(f"{'source':16s} {'G':>5s} {'real chi2/N':>11s} {'gaiamock':>9s}")
for r in rows.itertuples():
    if dr4.PRERELEASE_SOURCES[r.sid][1] == "orbit":
        continue
    d = eps[r.sid].as_jax(); _, chi2, _ = E.fit_single_star(d); n = eps[r.sid].n
    lin, _, _ = E.fit_single_star(d); plx, pmra, pmdec = float(lin[2]), float(lin[3]), float(lin[4])
    vals = []
    for k in range(10):
        truth = dict(period=np.array([1000.0]), ecc=np.array([0.0]), a0_mas=np.array([0.0]), inc=np.array([1.0]), Omega=np.array([0.0]), omega=np.array([0.0]),
                     t_peri=np.array([0.0]), parallax=np.array([plx]), pmra=np.array([pmra]), pmdec=np.array([pmdec]))
        ep = GB.simulate(rng, truth, r.ra0, r.dec0, r.G, "dr4")[0]
        _, c2, _ = E.fit_single_star(ep.as_jax()); vals.append(float(c2) / (ep.n - 5))
    print(f"{r.name:16s} {r.G:5.2f} {float(chi2)/(n-5):11.2f} {np.median(vals):9.2f}")

print("\nD. Gaia BH3: real fit vs gaiamock realisations of the same orbit at BH3's position and G")
sid = 4318465066420528000; ep = eps[sid]; d = ep.as_jax(); periods = np.geomspace(500, 8000, 400)
ll, be, bp = E.period_search(d, periods); k = int(np.argmax(np.asarray(ll)))
r = E.refine_profile(d, periods[k], float(be[k]), float(bp[k])); lin = r["linear"]
sig_plx_real = float(np.sqrt(r["linear_cov"][2, 2]))
red = r["chi2"] / (ep.n - 12)
print(f"  real ({ep.n} CCD abscissae, {used[used.source_id==sid].transit_id.nunique()} transits): P {r['period']:.0f} d, e {r['ecc']:.3f}, a0 {r['a0_mas']:.2f} mas, i {np.degrees(r['inc']):.1f}, plx {lin['parallax']:.4f} +- {sig_plx_real:.4f}, "
      f"significance {r['significance']:.0f}, chi2/dof {red:.2f}")
print(f"       the reported per-CCD errors ({float(used[used.source_id==sid].err.median()):.3f} mas) underestimate the scatter by sqrt({red:.2f}) = {np.sqrt(red):.2f}: "
      f"effective error {float(used[used.source_id==sid].err.median())*np.sqrt(red):.3f} mas, plx +- {sig_plx_real*np.sqrt(red):.4f}, significance {r['significance']/np.sqrt(red):.0f}")
truth = {k: np.array([v]) for k, v in dict(period=r["period"], ecc=r["ecc"], a0_mas=r["a0_mas"], inc=r["inc"], Omega=r["Omega"], omega=r["omega"], t_peri=r["t_peri"],
                                            parallax=lin["parallax"], pmra=lin["pmra"], pmdec=lin["pmdec"]).items()}
row = rows[rows.sid == sid].iloc[0]; rec = []
for k in range(20):
    epm = GB.simulate(rng, truth, row.ra0, row.dec0, row.G, "dr4")[0]; dm = epm.as_jax()
    llm, bem, bpm = E.period_search(dm, periods); km = int(np.argmax(np.asarray(llm)))
    rm = E.refine_profile(dm, periods[km], float(bem[km]), float(bpm[km]))
    rec.append([epm.n, rm["period"], rm["ecc"], rm["a0_mas"], np.degrees(rm["inc"]), rm["linear"]["parallax"], np.sqrt(rm["linear_cov"][2, 2]), rm["significance"], rm["chi2"] / (epm.n - 12)])
rec = np.array(rec); med = np.median(rec, axis=0); sd = rec.std(axis=0)
print(f"  gaiamock x20 ({med[0]:.0f} transit abscissae): P {med[1]:.0f} +- {sd[1]:.0f} d, e {med[2]:.3f} +- {sd[2]:.3f}, a0 {med[3]:.2f} +- {sd[3]:.2f} mas, i {med[4]:.1f} +- {sd[4]:.1f}, "
      f"plx {med[5]:.4f} +- {sd[5]:.4f} (formal {med[6]:.4f}), significance {med[7]:.0f} +- {sd[7]:.0f}, chi2/dof {med[8]:.2f}")
print("  (+- are the scatter over realisations; the real data's formal parallax error is listed above)")
