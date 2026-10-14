"""build_facts.py — the ONLY place numbers enter the film.

One example system, computed with darkcomp (the same model the pipeline fits) and gaiamock (the scan times,
angles and errors Gaia would have for this sky position in DR4). Every number the narration or the screen
uses is asserted here and written to facts.json; scenes.js reads nothing else.

    python build_facts.py
"""
import hashlib, json, os, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("DARKCOMP_PRECISION", "64")
import jax.numpy as jnp  # noqa: E402
import darkcomp  # noqa: E402,F401
from darkcomp import epoch as E, orbit, mock, gaiamock_bridge as GB  # noqa: E402
from darkcomp.constants import GM_SUN_AU3_DAY2  # noqa: E402

sources = {}
def src(path):
    p = pathlib.Path(path); sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest(); return p
src(REPO / "darkcomp" / "orbit.py"); src(REPO / "darkcomp" / "epoch.py"); src(REPO / "darkcomp" / "gaiamock_bridge.py")

# ---- the example system (same as docs/figures/astrometric_tracks.png, column "M2 = 1.4 Msun")
M_STAR, M_2, P_DAY, ECC, INC_DEG, DIST_PC, G_MAG = 1.0, 1.4, 600.0, 0.3, 60.0, 500.0, 14.0
OMEGA_BIG, OMEGA_SMALL, T_PERI, PM = 0.8, 2.3, 100.0, (8.0, -5.0)
rng = np.random.default_rng(1)
truth = mock.physical_population(rng, 1, m_star=M_STAR, m_2=M_2, dist_pc=DIST_PC, period=P_DAY)
truth["ecc"][:] = ECC; truth["inc"][:] = np.radians(INC_DEG); truth["Omega"][:] = OMEGA_BIG; truth["omega"][:] = OMEGA_SMALL
truth["t_peri"][:] = T_PERI; truth["pmra"][:] = PM[0]; truth["pmdec"][:] = PM[1]; truth["d_ra_mas"][:] = 0; truth["d_dec_mas"][:] = 0
plx = float(truth["parallax"][0]); a0 = float(truth["a0_mas"][0])
a_au = float(orbit.semimajor_axis_au(P_DAY, M_STAR, M_2)); q = M_2 / M_STAR
assert abs(plx - 1000.0 / DIST_PC) < 1e-9
assert abs(a0 - a_au * plx * q / (1 + q)) < 1e-9, "a0 = a * parallax * q/(1+q) for a dark companion"
assert abs(a_au ** 3 - GM_SUN_AU3_DAY2 * (M_STAR + M_2) * P_DAY ** 2 / (4 * np.pi ** 2)) < 1e-9, "Kepler's third law"
assert 2.1 < a0 < 2.25, f"narration says 'two point two milliarcseconds', a0 = {a0:.3f}"
assert abs(plx - 2.0) < 1e-9, "narration says 'two milliarcseconds' for the parallax"
assert abs(a_au - 1.864) < 0.01, f"narration says 'one point nine astronomical units', a = {a_au:.3f}"

# ---- gaiamock: scan epochs, angles, parallax factors, abscissae and errors for a random sky position
ra, dec = GB.random_sky(rng, 1)
ep = GB.simulate(np.random.default_rng(1), truth, ra, dec, np.full(1, G_MAG), "dr4")[0]
keep = ep.mask > 0; ts = ep.t_day[keep]; psi = ep.psi[keep]; w = ep.w[keep]; err = ep.err[keep]; plxf = ep.plx_factor[keep]
n_tr = int(keep.sum()); span_yr = float((ts.max() - ts.min()) / 365.25)
assert 50 <= n_tr <= 150, n_tr
assert 5.0 < span_yr < 5.6, span_yr
assert abs(float(err[0]) - 0.053) < 0.002, f"narration says 'five hundredths of a milliarcsecond', err = {err[0]:.4f}"

# ---- dense tracks over the span (days relative to J2017.5)
t = np.linspace(ts.min(), ts.max(), 1600)
X, Y = (np.asarray(x) for x in E.keplerian_xy(jnp.asarray(t), P_DAY, ECC, 2 * np.pi * T_PERI / P_DAY))
A, B, F, G_ = (float(x) for x in orbit.thiele_innes_ABFG(a0, OMEGA_SMALL, OMEGA_BIG, np.radians(INC_DEG)))
star_ra, star_dec = B * X + G_ * Y, A * X + F * Y                 # photocentre = the star (dark companion)
comp_ra, comp_dec = -star_ra / q, -star_dec / q                   # companion orbit around the centre of mass
assert abs(np.hypot(star_ra, star_dec).max() - a0 * (1 + ECC) * 0.98) < a0 * 0.3   # apastron distance order of magnitude (projected)

from astropy.coordinates import get_body_barycentric  # noqa: E402
from astropy.time import Time  # noqa: E402
import astropy.units as u  # noqa: E402
def parallax_offsets(t_day):
    tt = Time(t_day + E.DR4_REF_JD, format="jd", scale="tcb")
    s = get_body_barycentric("earth", tt).xyz.to(u.au).value
    ra_r, dec_r = np.radians(ra[0]), np.radians(dec[0])
    e_ra = np.array([-np.sin(ra_r), np.cos(ra_r), 0.0]); e_dec = np.array([-np.cos(ra_r) * np.sin(dec_r), -np.sin(ra_r) * np.sin(dec_r), np.cos(dec_r)])
    return -plx * (e_ra @ s), -plx * (e_dec @ s), s
par_ra, par_dec, earth = parallax_offsets(t)
assert abs(np.hypot(par_ra, par_dec).max() - plx) < 0.02 * plx, "parallax ellipse semi-major axis = parallax"
# the along-scan parallax factor from gaiamock should match the projection of our 2-D ellipse
pr_s, pd_s, _ = parallax_offsets(ts)
assert np.abs((pr_s * np.sin(psi) + pd_s * np.cos(psi)) / plx - plxf).max() < 0.03, "2-D parallax ellipse vs gaiamock AL parallax factors"
t_yr = t / 365.25
pm_ra, pm_dec = PM[0] * t_yr, PM[1] * t_yr
pm_total = float(np.hypot(*PM) * span_yr)
assert 48 < pm_total < 55, f"narration says 'about fifty milliarcseconds', pm displacement = {pm_total:.1f}"

# the measured abscissa must equal the model projection + noise at the scan epochs
Xs, Ys = (np.asarray(x) for x in E.keplerian_xy(jnp.asarray(ts), P_DAY, ECC, 2 * np.pi * T_PERI / P_DAY))
w_model = (PM[0] * ts / 365.25 + B * Xs + G_ * Ys) * np.sin(psi) + (PM[1] * ts / 365.25 + A * Xs + F * Ys) * np.cos(psi) + plx * plxf
resid = w - w_model
assert np.std(resid) < 2.5 * err[0] and np.abs(resid).max() < 5 * err[0], "abscissae are model + Gaussian noise of the quoted error"

# derived mass, the way the pipeline does it
f_m = (a0 / plx) ** 3 / (P_DAY / 365.25) ** 2
m2_check = float(__import__("darkcomp.batch", fromlist=["m2_from_mass_function"]).m2_from_mass_function(jnp.asarray(f_m), M_STAR))
assert abs(m2_check - M_2) < 1e-3, m2_check

r = lambda x: np.round(np.asarray(x, float), 4).tolist()
json.dump(dict(
    system=dict(m_star=M_STAR, m_2=M_2, period_day=P_DAY, ecc=ECC, inc_deg=INC_DEG, dist_pc=DIST_PC, g_mag=G_MAG, parallax_mas=plx,
                a_au=round(a_au, 4), a0_mas=round(a0, 4), q=q, pm_mas_yr=list(PM), pm_total_mas=round(pm_total, 2), span_yr=round(span_yr, 3),
                n_transits=n_tr, err_mas=round(float(err[0]), 4), ra_deg=round(float(ra[0]), 3), dec_deg=round(float(dec[0]), 3),
                year0=2017.5),
    t_day=r(t), star=dict(ra=r(star_ra), dec=r(star_dec)), companion=dict(ra=r(comp_ra), dec=r(comp_dec)),
    parallax=dict(ra=r(par_ra), dec=r(par_dec)), earth_au=dict(x=r(earth[0]), y=r(earth[1]), z=r(earth[2])),
    pm=dict(ra=r(pm_ra), dec=r(pm_dec)),
    scans=dict(t_day=r(ts), psi=r(psi), w=r(w), err=r(err), resid=r(resid), plx_factor=r(plxf)),
    sources=sources), open(HERE / "facts.json", "w"))
print(f"facts.json ok | a = {a_au:.3f} au, a0 = {a0:.3f} mas, plx = {plx:.2f} mas, {n_tr} transits over {span_yr:.2f} yr, "
      f"err {err[0]:.3f} mas, pm displacement {pm_total:.1f} mas, resid rms {np.std(resid):.3f}")
