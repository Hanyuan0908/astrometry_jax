"""Figure: how the astrometric track and Gaia's along-scan residuals change with companion mass.

    python scripts/plot_tracks.py [--dist 500] [--period 600] [--g-mag 14] [--out docs/figures/astrometric_tracks.png]

Rows, for a single star and for dark companions of increasing mass around a 1 Msun primary:
1. full sky path of the photocentre over the DR4 span: proper motion + parallax + orbit, with the
   proper motion recovered by the single-star fit and by the orbit fit (an unmodelled orbit biases it);
2. the same with proper motion removed, plus the observations: Gaia measures one coordinate per transit,
   along the scan direction, so each measurement is drawn as a bar through the position at that epoch,
   oriented along the scan angle, of length +- the error, offset by the measured residual;
3. what the pipeline sees: along-scan residuals after the best single-star (5-parameter) fit at gaiamock's
   DR4 scan times for a random sky position with gaiamock's per-transit error at the given G (step 1);
4. the periodogram: chi2 improvement over the single-star fit vs trial period, maximised over eccentricity
   and phase at each period (step 2), with the true period marked;
5. residuals after the Levenberg-Marquardt orbital fit started at the periodogram peak (step 3).
"""
import argparse, os, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, jax.numpy as jnp
import darkcomp
from darkcomp import epoch as E, orbit, mock, gaiamock_bridge as GB
from darkcomp.batch import mass_function_astrometric, m2_from_mass_function

ap = argparse.ArgumentParser()
ap.add_argument("--dist", type=float, default=500.0); ap.add_argument("--period", type=float, default=600.0)
ap.add_argument("--ecc", type=float, default=0.3); ap.add_argument("--inc", type=float, default=60.0)
ap.add_argument("--g-mag", type=float, default=14.0); ap.add_argument("--m-star", type=float, default=1.0)
ap.add_argument("--m2", type=float, nargs="+", default=[0.0, 0.6, 1.4, 3.0, 10.0])
ap.add_argument("--pm", type=float, nargs=2, default=[8.0, -5.0], help="pmra pmdec [mas/yr]")
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", default="docs/figures/astrometric_tracks.png")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
plx = 1000.0 / args.dist
n = len(args.m2)
truth = mock.physical_population(rng, n, m_star=args.m_star, m_2=np.maximum(args.m2, 1e-9), dist_pc=args.dist, period=args.period)
truth["ecc"][:] = args.ecc; truth["inc"][:] = np.radians(args.inc); truth["Omega"][:] = 0.8; truth["omega"][:] = 2.3; truth["t_peri"][:] = 100.0
truth["pmra"][:] = args.pm[0]; truth["pmdec"][:] = args.pm[1]; truth["d_ra_mas"][:] = 0.0; truth["d_dec_mas"][:] = 0.0
truth["a0_mas"][np.asarray(args.m2) == 0] = 0.0
ra, dec = GB.random_sky(rng, 1); ra, dec = np.repeat(ra, n), np.repeat(dec, n)
gm, c = GB.load()
tab = gm.get_gost_one_position(float(ra[0]), float(dec[0]), "dr4")
jd_scan = np.asarray(gm.fetch_table_element("ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]", tab))
# dense time grid for the sky path, and the scan epochs' positions
t_dense = np.linspace(jd_scan.min(), jd_scan.max(), 2000) - E.DR4_REF_JD
# parallax ellipse in the tangent plane needs the sun's position: approximate with the along-scan factor's generator;
# for the picture, use the standard ecliptic approximation at this sky position.
from astropy.coordinates import SkyCoord, get_body_barycentric
from astropy.time import Time
import astropy.units as u
sc = SkyCoord(ra[0] * u.deg, dec[0] * u.deg)
def parallax_offsets(t_day):
    tt = Time(t_day + E.DR4_REF_JD, format="jd", scale="tcb")
    s = get_body_barycentric("earth", tt).xyz.to(u.au).value  # (3, N) au, ICRS
    ra_r, dec_r = np.radians(ra[0]), np.radians(dec[0])
    e_ra = np.array([-np.sin(ra_r), np.cos(ra_r), 0.0]); e_dec = np.array([-np.cos(ra_r) * np.sin(dec_r), -np.sin(ra_r) * np.sin(dec_r), np.cos(dec_r)])
    # parallactic displacement = -plx * (projection of the observer's barycentric position)
    return -plx * (e_ra @ s), -plx * (e_dec @ s)
pra, pdec = parallax_offsets(t_dense)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(5, n, figsize=(3.3 * n, 15.0))
periods = np.geomspace(20, 3000, 300)
sims = GB.simulate(np.random.default_rng(args.seed), truth, ra, dec, np.full(n, args.g_mag), "dr4")
for j, m2 in enumerate(args.m2):
    p = {k: float(v[j]) for k, v in truth.items()}
    A, B, F, G = (float(x) for x in orbit.thiele_innes_ABFG(p["a0_mas"], p["omega"], p["Omega"], p["inc"]))
    X, Y = (np.asarray(x) for x in E.keplerian_xy(jnp.asarray(t_dense), p["period"], p["ecc"], 2 * np.pi * p["t_peri"] / p["period"]))
    ora, odec = B * X + G * Y, A * X + F * Y
    t_yr = t_dense / 365.25
    ep = sims[j]; keep = ep.mask > 0; ts = ep.t_day[keep]; psi_s = ep.psi[keep]; w_obs = ep.w[keep]; err_s = ep.err[keep]
    d = ep.as_jax(); beta, chi2, _ = E.fit_single_star(d)
    # model position at the scan epochs and the measured along-scan residual relative to the truth
    Xs, Ys = (np.asarray(x) for x in E.keplerian_xy(jnp.asarray(ts), p["period"], p["ecc"], 2 * np.pi * p["t_peri"] / p["period"]))
    pra_s, pdec_s = parallax_offsets(ts)
    ora_s, odec_s = B * Xs + G * Ys, A * Xs + F * Ys
    w_model = (p["pmra"] * ts / 365.25 + ora_s) * np.sin(psi_s) + (p["pmdec"] * ts / 365.25 + odec_s) * np.cos(psi_s) + plx * ep.plx_factor[keep]
    resid = w_obs - w_model
    # the orbital fit (steps 2 and 3 of the pipeline), needed here to overlay the fitted track
    ll, be, bp = E.period_search(d, periods); ll = np.asarray(ll); k = int(np.argmax(ll))
    dchi2 = 2 * (ll - (-0.5 * float(chi2)))
    r = E.refine_profile(d, periods[k], float(be[k]), float(bp[k])); lf = r["linear"]
    M = E.orbital_design(d, r["period"], r["ecc"], r["phase"]); beta9, chi2o, _ = E.weighted_lstsq(M, d)
    Xf, Yf = (np.asarray(x) for x in E.keplerian_xy(jnp.asarray(t_dense), r["period"], r["ecc"], r["phase"]))
    fra, fdec = lf["B"] * Xf + lf["G"] * Yf, lf["A"] * Xf + lf["F"] * Yf          # fitted orbit
    sra, sdec = lf["d_ra_mas"] + (lf["parallax"] / plx) * pra, lf["d_dec_mas"] + (lf["parallax"] / plx) * pdec   # fitted offsets + parallax
    # row 1: full path with proper motion
    a = ax[0, j]
    a.plot(p["pmra"] * t_yr + pra + ora, p["pmdec"] * t_yr + pdec + odec, "-", lw=0.6, c="C0")
    a.plot(0, 0, "k+", ms=8); a.set_aspect("equal"); a.invert_xaxis()
    pm1 = (float(beta[3]), float(beta[4]))
    a.set_title(("single star" if m2 == 0 else f"M2 = {m2:g} Msun, a0 = {p['a0_mas']:.2f} mas") +
                f"\npm fit (5-par): {pm1[0]:.2f}, {pm1[1]:.2f}  true {p['pmra']:g}, {p['pmdec']:g} mas/yr", fontsize=9)
    a.set_xlabel("RA offset [mas]")
    if j == 0:
        a.set_ylabel("Dec offset [mas]\n(with proper motion)")
    # row 2: proper motion removed, with the 1-D measurements
    a = ax[1, j]
    a.plot(pra + ora, pdec + odec, "-", lw=0.7, c="C0", label="photocentre path")
    a.plot(ora, odec, "-", lw=0.9, c="C3", alpha=0.8, label="true orbit")
    a.plot(sra + fra, sdec + fdec, "--", lw=0.7, c="k", alpha=0.7, label="fitted path (orbit fit)")
    a.plot(lf["d_ra_mas"] + fra, lf["d_dec_mas"] + fdec, "--", lw=0.9, c="C2", label="fitted orbit")
    xo, yo = pra_s + ora_s + resid * np.sin(psi_s), pdec_s + odec_s + resid * np.cos(psi_s)
    a.plot([xo - err_s * np.sin(psi_s), xo + err_s * np.sin(psi_s)], [yo - err_s * np.cos(psi_s), yo + err_s * np.cos(psi_s)], "-", c="0.3", lw=0.8)
    a.plot(xo, yo, ".", c="k", ms=3, label="along-scan measurements")
    a.plot(0, 0, "k+", ms=8); a.set_aspect("equal"); a.invert_xaxis()
    a.set_title(f"pm removed; plx fit {lf['parallax']:.3f} (true {plx:.2f})", fontsize=9); a.set_xlabel("RA offset [mas]")
    if j == 0:
        a.set_ylabel("Dec offset [mas]"); a.legend(fontsize=7, loc="upper left")
    # bottom: along-scan residuals after a single-star fit, at gaiamock scan times
    res = np.asarray(d["w"] - E.single_star_design(d) @ beta)[ep.mask > 0]
    b = ax[2, j]
    b.errorbar(ep.t_day[ep.mask > 0] / 365.25 + 2017.5, res, yerr=ep.err[ep.mask > 0], fmt=".", ms=3, lw=0.6, c="k")
    b.axhline(0, c="0.6", lw=0.6)
    b.set_title(f"single-star fit: chi2/N = {float(chi2)/ep.n:.1f}", fontsize=9)
    b.set_xlabel("year")
    if j == 0:
        b.set_ylabel("along-scan residual [mas]")
    # periodogram (step 2) and orbital fit (step 3)
    cpg = ax[3, j]; cpg.semilogx(periods, dchi2, "-", lw=0.8, c="k")
    if m2 > 0:
        cpg.axvline(p["period"], c="C3", lw=0.8, ls="--")
    cpg.set_title(f"periodogram: peak at P = {periods[k]:.0f} d, dchi2 = {dchi2[k]:.0f}", fontsize=9); cpg.set_xlabel("trial period [d]")
    if j == 0:
        cpg.set_ylabel("chi2(single) - chi2(orbit)")
    res_o = np.asarray(d["w"] - M @ beta9)[ep.mask > 0]
    cf = ax[4, j]; cf.errorbar(ep.t_day[ep.mask > 0] / 365.25 + 2017.5, res_o, yerr=ep.err[ep.mask > 0], fmt=".", ms=3, lw=0.6, c="k"); cf.axhline(0, c="0.6", lw=0.6)
    cf.set_title((f"orbit fit: P = {r['period']:.0f} d, a0 = {r['a0_mas']:.2f} mas, " if m2 > 0 else f"orbit fit (noise): a0 = {r['a0_mas']:.2f} mas, ") + f"chi2/N = {float(chi2o)/ep.n:.2f}\npm {r['linear']['pmra']:.2f}, {r['linear']['pmdec']:.2f}; significance {r['significance']:.1f} sigma, M2 = {float(m2_from_mass_function(mass_function_astrometric(r['a0_mas'], r['linear']['parallax'], r['period']), args.m_star)):.2f} Msun", fontsize=7)
    cf.set_xlabel("year")
    if j == 0:
        cf.set_ylabel("residual after orbit fit [mas]")
ylim = max(np.abs(np.concatenate([l.get_ydata() for a_ in ax[2] for l in a_.get_lines()])).max(), 0.2)
for b in list(ax[2]) + list(ax[4]):
    b.set_ylim(-ylim * 1.1, ylim * 1.1)
fig.suptitle(f"{args.m_star:g} Msun star at {args.dist:g} pc (parallax {plx:.1f} mas), G = {args.g_mag:g}, P = {args.period:g} d, e = {args.ecc:g}, i = {args.inc:g} deg; "
             f"DR4 scanning law, {int(ep.n)} transits; proper motion {args.pm[0]:g}, {args.pm[1]:g} mas/yr", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.975]); plt.savefig(args.out, dpi=140); print("saved", args.out)
