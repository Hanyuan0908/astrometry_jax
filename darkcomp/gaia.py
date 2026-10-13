"""Gaia DR3 nss_two_body_orbit access and covariance reconstruction.

Parameter ordering in the archive (and in `corr_vec`) for the two solution types
we support -- this is also the ordering of the predicted vector theta_Gaia in the
likelihood (El-Badry et al. 2024 eqs. 11-12):

  Orbital          : ra, dec, parallax, pmra, pmdec, A, B, F, G, e, P, t_peri          (12)
  AstroSpectroSB1  : ra, dec, parallax, pmra, pmdec, A, B, F, G, C, H, gamma, e, P, t_peri (15)

`corr_vec` holds the strict lower triangle of the correlation matrix, row by row:
(1,0), (2,0), (2,1), (3,0), ... .  ra/dec errors are in mas (ra_error is for ra*cos(dec));
we therefore work with sky offsets d_ra* and d_dec in mas relative to the archive position.
"""
from dataclasses import dataclass, field
import json
import numpy as np

ORBITAL_COLS = ["ra", "dec", "parallax", "pmra", "pmdec",
                "a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes",
                "eccentricity", "period", "t_periastron"]
SB1_COLS = ["ra", "dec", "parallax", "pmra", "pmdec",
            "a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes",
            "c_thiele_innes", "h_thiele_innes", "center_of_mass_velocity",
            "eccentricity", "period", "t_periastron"]

COLS_BY_TYPE = {"Orbital": ORBITAL_COLS, "AstroSpectroSB1": SB1_COLS}

NSS_QUERY_COLS = sorted(set(SB1_COLS) | {c + "_error" for c in SB1_COLS} |
                        {"source_id", "nss_solution_type", "corr_vec", "goodness_of_fit",
                         "bit_index", "astrometric_n_obs_al"})


@dataclass
class GaiaSolution:
    source_id: int
    solution_type: str              # "Orbital" or "AstroSpectroSB1"
    names: list                     # parameter names in archive order
    mean: np.ndarray                # (n,) best-fit values; ra, dec replaced by 0 mas offsets
    cov: np.ndarray                 # (n, n) covariance in working units
    ra_deg: float                   # reference position for the mas offsets
    dec_deg: float
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.names)

    def __getitem__(self, name):
        return float(self.mean[self.names.index(name)])

    def sigma(self, name):
        i = self.names.index(name)
        return float(np.sqrt(self.cov[i, i]))

    @property
    def is_sb1(self):
        return self.solution_type == "AstroSpectroSB1"


def corr_matrix_from_vec(corr_vec, n):
    """Full symmetric correlation matrix from the Gaia lower-triangular row-major vector."""
    corr_vec = np.asarray(corr_vec, dtype=float)
    assert corr_vec.size == n * (n - 1) // 2, (corr_vec.size, n)
    C = np.eye(n)
    k = 0
    for i in range(1, n):
        for j in range(i):
            C[i, j] = C[j, i] = corr_vec[k]
            k += 1
    return C


def solution_from_row(row):
    """Build a GaiaSolution from a dict-like archive row (astropy Row, pandas Series, dict)."""
    get = lambda k: row[k]  # noqa: E731
    stype = str(get("nss_solution_type"))
    if isinstance(stype, bytes):
        stype = stype.decode()
    if stype not in COLS_BY_TYPE:
        raise ValueError(f"unsupported nss_solution_type {stype!r}; need Orbital or AstroSpectroSB1")
    cols = COLS_BY_TYPE[stype]
    n = len(cols)
    mean = np.array([float(get(c)) for c in cols])
    err = np.array([float(get(c + "_error")) for c in cols])
    corr = corr_matrix_from_vec(get("corr_vec"), n)
    cov = corr * np.outer(err, err)
    ra_deg, dec_deg = mean[0], mean[1]
    mean = mean.copy()
    mean[0] = 0.0  # d_ra* [mas]
    mean[1] = 0.0  # d_dec [mas]
    names = ["d_ra_mas", "d_dec_mas"] + cols[2:]
    meta = {k: row[k] for k in ("goodness_of_fit", "bit_index", "astrometric_n_obs_al") if k in _keys(row)}
    return GaiaSolution(int(get("source_id")), stype, names, mean, cov, float(ra_deg), float(dec_deg), meta)


def _keys(row):
    if hasattr(row, "colnames"):
        return row.colnames
    if hasattr(row, "keys"):
        return list(row.keys())
    return list(row.index)


def query_nss(source_ids):
    """Fetch nss_two_body_orbit rows for one or more source_ids (needs network + astroquery)."""
    from astroquery.gaia import Gaia

    ids = ",".join(str(int(s)) for s in np.atleast_1d(source_ids))
    q = f"SELECT {', '.join(NSS_QUERY_COLS)} FROM gaiadr3.nss_two_body_orbit WHERE source_id IN ({ids})"
    return Gaia.launch_job(q).get_results()


def load_solution(source_id, fixture_path=None):
    """GaiaSolution for `source_id`, from a JSON fixture if given else from the archive."""
    if fixture_path is not None:
        with open(fixture_path) as f:
            rows = json.load(f)
        for r in rows:
            if int(r["source_id"]) == int(source_id):
                return solution_from_row(r)
        raise KeyError(f"{source_id} not in {fixture_path}")
    tab = query_nss([source_id])
    if len(tab) == 0:
        raise KeyError(f"{source_id} has no nss_two_body_orbit solution")
    return solution_from_row(tab[0])
