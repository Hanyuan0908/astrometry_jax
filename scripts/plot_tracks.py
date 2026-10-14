"""Figure: how the astrometric track and Gaia's along-scan residuals change with companion mass.

    python scripts/plot_tracks.py [--dist 500] [--period 600] [--g-mag 14] [--out docs/figures/astrometric_tracks.png]

Rows, for a single star and for dark companions of increasing mass around a 1 Msun primary:
1. sky path of the photocentre over the DR4 span (proper motion removed, parallax ellipse kept);
2. what Gaia measures: along-scan residuals after the best single-star (5-parameter) fit, at gaiamock's
   DR4 scan times for a random sky position with gaiamock's per-transit error at the given G (pipeline step 1);
3. the periodogram: profile log-likelihood improvement over the single-star fit vs trial period, maximised
   over eccentricity and phase at each period (pipeline step 2), with the true period marked;
4. residuals after the Levenberg-Marquardt orbital fit started at the periodogram peak (pipeline step 3).
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
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", default="docs/figures/astrometric_tracks.png")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
plx = 1000.0 / args.dist
n = len(args.m2)
truth = mock.physical_population(rng, n, m_star=args.m_star, m_2=np.maximum(args.m2, 1e-9), dist_pc=args.dist, period=args.period)
truth["ecc"][:] = args.ecc; truth["inc"][:] = np.radians(args.inc); truth["Omega"][:] = 0.8; truth["omega"][:] = 2.3; truth["t_peri"][:] = 100.0
truth["pmra"][:] = 0.0; truth["pmdec"][:] = 0.0; truth["d_ra_mas"][:] = 0.0; truth["d_dec_mas"][:] = 0.0
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
fig, ax = plt.subplots(4, n, figsize=(3.3 * n, 12.0))
periods = np.geomspace(20, 3000, 300)
sims = GB.simulate(np.random.default_rng(args.seed), truth, ra, dec, np.full(n, args.g_mag), "dr4")
for j, m2 in enumerate(args.m2):
    p = {k: float(v[j]) for k, v in truth.items()}
    A, B, F, G = (float(x) for x in orbit.thiele_innes_ABFG(p["a0_mas"], p["omega"], p["Omega"], p["inc"]))
    X, Y = (np.asarray(x) for x in E.keplerian_xy(jnp.asarray(t_dense), p["period"], p["ecc"], 2 * np.pi * p["t_peri"] / p["period"]))
    ora, odec = B * X + G * Y, A * X + F * Y
    a = ax[0, j]
    a.plot(pra + ora, pdec + odec, "-", lw=0.7, c="C0", label="photocentre path")
    a.plot(ora, odec, "-", lw=0.9, c="C3", alpha=0.8, label="orbit only")
    a.plot(0, 0, "k+", ms=8)
    a.set_aspect("equal"); a.invert_xaxis()
    a.set_title(("single star" if m2 == 0 else f"M2 = {m2:g} Msun") + ("" if m2 == 0 else f"\na0 = {p['a0_mas']:.2f} mas"), fontsize=10)
    a.set_xlabel("RA offset [mas]"); 
    if j == 0:
        a.set_ylabel("Dec offset [mas]"); a.legend(fontsize=7, loc="upper left")
    # bottom: along-scan residuals after a single-star fit, at gaiamock scan times
    ep = sims[j]; d = ep.as_jax(); beta, chi2, _ = E.fit_single_star(d)
    res = np.asarray(d["w"] - E.single_star_design(d) @ beta)[ep.mask > 0]
    b = ax[1, j]
    b.errorbar(ep.t_day[ep.mask > 0] / 365.25 + 2017.5, res, yerr=ep.err[ep.mask > 0], fmt=".", ms=3, lw=0.6, c="k")
    b.axhline(0, c="0.6", lw=0.6)
    b.set_title(f"single-star fit: chi2/N = {float(chi2)/ep.n:.1f}", fontsize=9)
    b.set_xlabel("year")
    if j == 0:
        b.set_ylabel("along-scan residual [mas]")
    # periodogram (step 2) and orbital fit (step 3)
    ll, be, bp = E.period_search(d, periods); ll = np.asarray(ll); k = int(np.argmax(ll))
    dchi2 = 2 * (ll - (-0.5 * float(chi2)))
    cpg = ax[2, j]; cpg.semilogx(periods, dchi2, "-", lw=0.8, c="k")
    if m2 > 0:
        cpg.axvline(p["period"], c="C3", lw=0.8, ls="--")
    cpg.set_title(f"periodogram: peak at P = {periods[k]:.0f} d, dchi2 = {dchi2[k]:.0f}", fontsize=9); cpg.set_xlabel("trial period [d]")
    if j == 0:
        cpg.set_ylabel("chi2(single) - chi2(orbit)")
    r = E.refine_profile(d, periods[k], float(be[k]), float(bp[k]))
    M = E.orbital_design(d, r["period"], r["ecc"], r["phase"])
    beta9, chi2o, _ = E.weighted_lstsq(M, d)
    res_o = np.asarray(d["w"] - M @ beta9)[ep.mask > 0]
    cf = ax[3, j]; cf.errorbar(ep.t_day[ep.mask > 0] / 365.25 + 2017.5, res_o, yerr=ep.err[ep.mask > 0], fmt=".", ms=3, lw=0.6, c="k"); cf.axhline(0, c="0.6", lw=0.6)
    cf.set_title((f"orbit fit: P = {r['period']:.0f} d, a0 = {r['a0_mas']:.2f} mas, " if m2 > 0 else f"orbit fit (noise): a0 = {r['a0_mas']:.2f} mas, ") + f"chi2/N = {float(chi2o)/ep.n:.2f}\nsignificance {r['significance']:.1f} sigma, M2 = {float(m2_from_mass_function(mass_function_astrometric(r['a0_mas'], r['linear']['parallax'], r['period']), args.m_star)):.2f} Msun", fontsize=8)
    cf.set_xlabel("year")
    if j == 0:
        cf.set_ylabel("residual after orbit fit [mas]")
ylim = max(np.abs(np.concatenate([l.get_ydata() for a_ in ax[1] for l in a_.get_lines()])).max(), 0.2)
for b in list(ax[1]) + list(ax[3]):
    b.set_ylim(-ylim * 1.1, ylim * 1.1)
fig.suptitle(f"{args.m_star:g} Msun star at {args.dist:g} pc (parallax {plx:.1f} mas), G = {args.g_mag:g}, P = {args.period:g} d, e = {args.ecc:g}, i = {args.inc:g} deg; "
             f"DR4 scanning law, {int(ep.n)} transits; proper motion removed", fontsize=9)
plt.tight_layout(); plt.savefig(args.out, dpi=140); print("saved", args.out)
