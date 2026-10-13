"""Loader for Gaia DR4 `epoch_astrometry` tables (pre-release VOTable format, June 2026).

One VOTable row = one field-of-view transit, with per-CCD arrays (obs_time_tcb, scan_pos_angle,
centroid_pos_al, centroid_pos_error_al, used_by_agis_al, ...). We explode to one row per CCD
measurement and, following ESA's gaia-supdate tool, keep only `used_by_agis_al == True` and
`multipeak == False`. Times: obs_time_tcb is ns since 2010-01-01T00:00 TCB (JD 2455197.5);
obs_time_bary_corr (ns) is added. Reference epoch for DR4 astrometry is J2017.5.
"""
import io
import zipfile
import warnings
import numpy as np
import pandas as pd

from .epoch import EpochAstrometry, DR4_REF_JD

TCB_2010_JD = 2455197.5
NS_PER_DAY = 86400e9

PRERELEASE_SOURCES = {  # from https://www.cosmos.esa.int/web/gaia/dr4-prerelease
    4181040337841125632: ("HD 183633", "parallax"), 3926186255616949504: ("", "parallax"),
    435469040545191680: ("", "parallax"), 1663617687609809280: ("LSPM J1324+6124", "magnitude"),
    2309425390592896: ("", "magnitude"), 20694084440761600: ("", "magnitude"),
    2237987199365376: ("", "qso"), 60730287810150016: ("", "qso"), 10973744521070720: ("", "qso"),
    3937211745905473024: ("HD 114762", "orbit"), 1457486023639239296: ("Gaia-4", "orbit"),
    4318465066420528000: ("Gaia BH3", "orbit"),
}


def read_votable(path):
    """Return the exploded per-CCD pandas DataFrame for all sources in a VOTable (.xml or .zip)."""
    from astropy.io.votable import parse
    if str(path).endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.endswith(".xml")][0]
            raw = z.read(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tab = parse(io.BytesIO(raw)).get_first_table().to_table()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tab = parse(path).get_first_table().to_table()
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # astropy: masked (unused CCD) elements -> nan; dropped later
        for r in tab:
            n = len(np.atleast_1d(r["obs_time_tcb"]))
            rec = dict(source_id=int(r["source_id"]), transit_id=int(r["transit_id"]), ra0=float(r["ra0"]),
                       dec0=float(r["dec0"]), bary_corr_ns=float(r["obs_time_bary_corr"]),
                       plx_factor=float(r["parallax_factor_al"]), multipeak=bool(r["multipeak"]),
                       excess_noise=float(r["agis_source_excess_noise"]), g_mag=float(r["g_mag"]))
            for i in range(n):
                rows.append(dict(rec, ccd=i, t_ns=int(np.atleast_1d(r["obs_time_tcb"])[i]),
                                 psi_deg=float(np.atleast_1d(r["scan_pos_angle"])[i]),
                                 w=float(np.atleast_1d(r["centroid_pos_al"])[i]),
                                 err=float(np.atleast_1d(r["centroid_pos_error_al"])[i]),
                                 used_al=bool(np.atleast_1d(r["used_by_agis_al"])[i])))
    df = pd.DataFrame(rows)
    df["jd_tcb"] = TCB_2010_JD + (df["t_ns"] + df["bary_corr_ns"]) / NS_PER_DAY
    return df


def epoch_from_frame(df, source_id, filter_used=True, drop_multipeak=True, n_pad=None):
    s = df[df.source_id == int(source_id)]
    if filter_used:
        s = s[s.used_al]
    if drop_multipeak:
        s = s[~s.multipeak]
    s = s[np.isfinite(s.w) & np.isfinite(s.err)]
    return EpochAstrometry.from_arrays(s.jd_tcb, s.w, s.err, s.plx_factor, s.psi_deg, ref_jd=DR4_REF_JD,
                                      n_pad=n_pad, transit_id=s.transit_id.values)


def load_prerelease(path, **kw):
    """dict source_id -> EpochAstrometry for every source in the file."""
    df = read_votable(path)
    return {sid: epoch_from_frame(df, sid, **kw) for sid in sorted(df.source_id.unique())}, df
