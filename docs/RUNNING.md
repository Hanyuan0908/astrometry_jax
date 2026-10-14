# Running darkcomp

This guide covers installation, the precision switch, the tests, and the four ways the code is used:
a DR3 summary-solution fit, an epoch-astrometry fit of one source, the batched search over many sources,
and a NUTS posterior for a candidate. The GPU benchmark notebook is described at the end.

## 1. Install

Python 3.10 or newer. CPU-only JAX is enough for everything except the large batched searches.

```bash
git clone git@github.com:Hanyuan0908/astrometry_jax.git
cd astrometry_jax
pip install "jax[cpu]" numpy scipy pandas astropy astroquery numpyro jaxopt pytest matplotlib
```

With a conda environment (what the authors use):

```bash
conda create -n darkcomp python=3.12
conda activate darkcomp
pip install "jax[cpu]" numpy scipy pandas astropy astroquery numpyro jaxopt pytest matplotlib corner
```

For an NVIDIA GPU install `jax[cuda12]` (or whatever matches your CUDA) instead of `jax[cpu]`; on Google
Colab the preinstalled JAX already has GPU support and must not be upgraded.

Nothing needs to be `pip install`ed from this repository: the scripts and tests put the repository root
on `sys.path`. If you write your own script elsewhere, add

```python
import sys; sys.path.insert(0, "/path/to/astrometry_jax")
```

or run `pip install -e .` once from the repository root.

Network access is only needed for Gaia DR3 sources not already cached in `data/gaia_dr3_nss_fixture.json`
(the `astroquery` call to the Gaia archive). The tests never use the network.

## 2. Precision

The whole pipeline runs in one dtype, chosen once before JAX is imported:

```bash
DARKCOMP_PRECISION=64 python my_script.py     # default: float64
DARKCOMP_PRECISION=32 python my_script.py     # float32: use this on GPUs
```

Inside Python set the variable before the first `import darkcomp`:

```python
import os; os.environ["DARKCOMP_PRECISION"] = "32"
import darkcomp
```

JAX fixes the dtype on first use, so a running process cannot switch; restart it instead. float32 gives
the same detections and parameters as float64 to about 1e-5 relative, and is 2 to 60 times faster on GPUs
depending on the card. TF32 tensor-core matmuls are disabled by the package so that float32 is genuine.

## 3. Tests

```bash
python -m pytest -q                                   # 20 tests, ~30 s
DARKCOMP_PRECISION=32 python -m pytest -q             # the same suite in float32
python -m pytest -q tests/test_likelihood.py -k paper # one test by name pattern
```

The suite checks the Kepler solver, the orbital geometry, the DR3 likelihood against the published
J0553-1349 solution, the ensemble sampler, the BH3 epoch-astrometry solution, the DR4 pre-release
null and detected sources, and mock-orbit recovery through the batch pipeline. Run it after any change
to geometry or units.

## 4. DR3 summary-solution fit (El-Badry et al. 2024 method)

Fit one of the 21 paper sources with the bundled RVs and SED masses:

```bash
python scripts/fit_source.py J0553-1349                       # paper settings: 64 walkers, 3000 burn + 3000 steps, ~2 s
python scripts/fit_source.py J0553-1349 --no-rvs              # Gaia-only, as in the paper's comparison
python scripts/fit_source.py J2057-4742 --steps 500 --burn 500
```

The script prints the branch chosen, the MAP log-probability, the acceptance fraction, the autocorrelation
time, and the median with 16/84 percentiles of every fitted and derived quantity (a, a0, K, M2, mass function, ...).

Sources not in the fixture are fetched from the Gaia archive by `source_id` (see `data/elbadry2024_star_params.csv`
for the name to source_id mapping). To fit a source outside the paper sample:

```python
from darkcomp import gaia, data as D, fit
sol = gaia.load_solution(2995961897685517312)                 # queries the archive (nss_two_body_orbit)
rvs = D.RVData.from_arrays(jd, rv_kms, err_kms)                 # your own RVs, HJD/JD in days
res = fit.fit_source(sol, rvs, m_star_prior=(0.79, 0.04))      # (mean, sigma) in Msun
for k, (med, up, lo) in res.summary().items():
    print(f"{k:12s} {med:10.4f} +{up:.4f} -{lo:.4f}")
```

Conventions worth knowing: inclination is sampled in [0, 180] deg and not folded; RV sign convention is the
paper's; no parallax zero-point correction is applied.

## 5. Epoch astrometry: one source

### 5a. Gaia BH3 from the CDS table

```python
import numpy as np
from darkcomp import epoch as E
ep = E.EpochAstrometry.from_bh3_table("data/gaia_bh3/epochast.dat")   # 622 abscissae
d = ep.as_jax()
periods = np.geomspace(50, 8000, 600)
ll, be, bp = E.period_search(d, periods)             # profile log-likelihood on a (P, e, phase) grid, chunked
k = int(np.argmax(np.asarray(ll)))
r = E.refine_profile(d, periods[k], float(be[k]), float(bp[k]))
print(r["period"], r["ecc"], r["a0_mas"], np.degrees(r["inc"]), r["linear"]["parallax"], r["chi2"])
```

`refine_profile` returns the orbital elements (period, ecc, t_peri, a0_mas, inc, Omega, omega), the linear
parameters (`r["linear"]`: d_ra_mas, d_dec_mas, parallax, pmra, pmdec and the Thiele-Innes A, B, F, G), the
chi2 and the covariance. For bright stars whose per-CCD errors are underestimated pass `fit_jitter=True` to
add an excess-noise term iterated until chi2/dof is 1 (HD 114762 needs 0.35 mas).

### 5b. Gaia DR4 epoch_astrometry files

```python
from darkcomp import dr4
eps, df = dr4.load_prerelease("data/dr4_prerelease/gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip")
ep = eps[4318465066420528000]        # Gaia BH3; eps is {source_id: EpochAstrometry}
```

`load_prerelease` accepts the VOTable `.xml` or the `.zip` that contains it. Per-CCD arrays are exploded,
and only measurements with `used_by_agis_al` set and not flagged `multipeak` are kept (as in ESA's
gaia-supdate). Times are TCB nanoseconds since 2010.0 plus the barycentric correction. The same function
will read the DR4 catalogue files when they are released.

### 5c. Your own abscissae

```python
ep = E.EpochAstrometry.from_arrays(jd, w_mas, err_mas, parallax_factor_al, scan_pos_angle_deg)
```

The reference epoch defaults to J2017.5 (`E.DR4_REF_JD`); pass `ref_jd=` to change it.

## 6. Batched search over many sources (the candidate filter)

```python
import numpy as np, jax, jax.numpy as jnp
from darkcomp import batch, dr4

eps, _ = dr4.load_prerelease("data/dr4_prerelease/gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip")
sources = list(eps.values())                       # any list of EpochAstrometry, different lengths are fine
dd = batch.stack_epochs(sources)                   # padded pytree with a leading source axis
periods = np.geomspace(20, 3000, 300)

n_sources, n_obs = dd["w"].shape
source_chunk, chunk = batch.plan_chunks(n_obs, n_sources, mem_budget_gb=1.0, dtype_bytes=4)   # 1 GB on a laptop
out = batch.run_pipeline(dd, jnp.asarray(periods), chunk=chunk, source_chunk=source_chunk,
                         refine_chunk=min(max(4 * source_chunk, 64), n_sources), m_star=1.0)
jax.block_until_ready(out["period"])

sig = np.asarray(out["significance"])
for i, sid in enumerate(eps):
    print(sid, f"P={float(out['period'][i]):8.1f} d  e={float(out['ecc'][i]):.2f}  a0={float(out['a0_mas'][i]):.3f} mas"
               f"  sig={sig[i]:6.1f}  dchi2={float(out['dchi2'][i]):8.1f}  M2={float(out['m_2'][i]):.2f} Msun")
```

What the pipeline does per source: 5-parameter single-star fit; profile-likelihood periodogram over
300 periods x 4 eccentricities (e <= 0.6) x 8 phases; Levenberg-Marquardt refinement of the three
highest local maxima, keeping the lowest chi2; Campbell elements; detection significance (the
Thiele-Innes vector against zero through its covariance); astrometric mass function and companion
mass for the given `m_star`.

Output keys (one value per source): `period, ecc, t_peri, a0_mas, inc, Omega, omega, parallax, pmra,
pmdec, d_ra_mas, d_dec_mas, chi2, chi2_single, dchi2, n_obs, significance, parallax_single, f_m_ast, m_2,
periodogram`. A source with significance below about 5 and dchi2 below about 25 is consistent with a
single star; in the DR4 pre-release the nine single stars and quasars give significance 3 to 5. For such
non-detections the orbital parameters are noise fits and meaningless (the eccentricity typically sits at
its bound and `m_2` can be absurd); only `significance` and `dchi2` carry information. The detected
sources in the same file come out as Gaia-4 (42 sigma, P = 579 d), HD 114762 (231 sigma, P = 83.8 d) and
BH3 (P = 4184 d, a0 = 27.1 mas, M2 = 35 Msun for the default m_star = 1).

**Memory.** Each step builds `source_chunk x chunk` design matrices of shape `(n_obs, 9)`. Always size
them with `plan_chunks` from the memory you are willing to use (0.6 x device memory on a GPU, 1 GB on a
laptop). A 480,000-point `vmap` once froze a 24 GB laptop; never call the unchunked functions on more
than a few hundred trial orbits at once.

**Throughput.** On an A100 the pipeline runs at about 124 sources/s for batches of 1000 or more; on an
L4 44 sources/s; on a laptop CPU 4 to 6 sources/s. Results are identical across devices and chunkings.

## 6a. Mock epoch astrometry with gaiamock

gaiamock (El-Badry et al. 2024, arXiv:2411.00088, https://github.com/kareemelbadry/gaiamock) is the
community tool for simulating Gaia epoch astrometry and is the preferred mock backend. It predicts the
scan times, angles and parallax factors at any sky position from the nominal scanning law, uses the
G-dependent along-scan uncertainty, drops a random 10% of transits and adds extra noise for G < 13.
`darkcomp.gaiamock_bridge` calls gaiamock's own `predict_astrometry_binary_in_terms_of_a0` and wraps
the result as `EpochAstrometry`, so the pipeline sees exactly what gaiamock produces.

One-time setup (about 1 GB on disk; keep it outside synced folders):

```bash
mkdir -p ~/software && cd ~/software
git clone https://github.com/kareemelbadry/gaiamock.git && cd gaiamock
# scanning-law files (286 MB zip, 984 MB unpacked): the Box link from the gaiamock README, or directly:
curl -L -A "Mozilla/5.0" -o healpix_scans.zip "https://caltech.box.com/index.php?rm=box_download_shared_file&shared_name=4f7q6qdh0bku881bzvzxc4cm5u0902cf&file_id=f_1611600757604"
unzip -q healpix_scans.zip && rm healpix_scans.zip          # -> healpix_scans/ with 49,152 FITS files
conda install -c conda-forge gsl healpy                      # or brew install gsl; pip install healpy
ENV=$(python -c "import sys; print(sys.prefix)")
gcc -shared -fPIC -o kepler_solve_astrometry.so kepler_solve_astrometry.c -I$ENV/include -L$ENV/lib -lgsl -lgslcblas -lm -Wl,-rpath,$ENV/lib
```

Point `GAIAMOCK_DIR` at the directory if it is not `~/software/gaiamock`. Then:

```python
from darkcomp import mock, gaiamock_bridge as GB
rng = np.random.default_rng(0)
truth = mock.physical_population(rng, 100, m_star=1.0, m_2=1.4, dist_pc=1000.0)   # a0 from the physics
ra, dec = GB.random_sky(rng, 100)
g_mag = GB.g_from_distance(4.5, truth["dist_pc"])              # M_G = 4.5: solar-type primary
eps = GB.simulate(rng, truth, ra, dec, g_mag, data_release="dr4")  # list of EpochAstrometry
```

gaiamock returns one measurement per field-of-view transit (8 CCDs averaged, error / sqrt(8)), 60 to
150 transits per source for DR4. On four test systems gaiamock's own `fit_full_astrometric_cascade`
and our pipeline return identical P, e, a0, parallax and inclination to the printed precision.
`python scripts/benchmark_gaiamock.py` compares gaiamock with the real DR4 pre-release data (transit
counts and times, scan angles, parallax factors, error model, single-star chi2, and a BH3 injection and
recovery); the outcome is summarised in the README.

## 6b. Detectability versus companion mass and distance

`mock.physical_population` builds mock systems from physical parameters (primary mass, dark-companion mass,
period, distance); a0 then follows from Kepler's law and the parallax. `scripts/detectability.py` runs a
grid of (M2, distance) cells through the batch pipeline:

```bash
# gaiamock backend (default when installed): random sky positions, G = M_G + 5 log10(d/10) with M_G = 4.5
DARKCOMP_PRECISION=32 python scripts/detectability.py --m2 0.1 0.3 0.6 1 1.4 3 10 --dist 500 1000 2000 5000 10000 --n 20
DARKCOMP_PRECISION=32 python scripts/detectability.py ... --g-mag 15             # fixed apparent magnitude instead
# template backend: real pre-release scan patterns and errors, optionally scaled
DARKCOMP_PRECISION=32 python scripts/detectability.py --backend templates ... --noise-scale 5
```

It prints the detection fraction per cell (and the fraction with the period recovered to 5%), the median
recovered M2, and writes `<out>.npz` and `<out>.png`. 700 sources take about 2.5 minutes on a laptop CPU
and a few seconds on a GPU. Results are in `docs/figures/detectability_gaiamock.*` (gaiamock, solar-type primary),
`detectability_bright.*` and `detectability_noise5.*` (templates). Options: `--m-star`, `--abs-mag`,
`--p-range`, `--e-max`, `--sig` (detection threshold), `--data-release`, `--all-templates`.

## 7. Posterior for a candidate (NUTS)

After the batch search, sample the full 14-parameter posterior of an interesting source, with or
without RVs:

```python
import numpy as np, pandas as pd
from darkcomp import epoch as E, data as D, nuts, fit

ep = E.EpochAstrometry.from_bh3_table("data/gaia_bh3/epochast.dat")
rv = pd.read_csv("data/gaia_bh3/epochrv.dat", names="transit_id jd rv err".split(), sep=r"\s+", comment="#")
rvd = D.RVData.from_arrays(rv.jd, rv.rv, rv.err)

# start from the profile-likelihood solution (Section 5a)
d = ep.as_jax(); periods = np.geomspace(50, 8000, 600)
ll, be, bp = E.period_search(d, periods); k = int(np.argmax(np.asarray(ll)))
r = E.refine_profile(d, periods[k], float(be[k]), float(bp[k])); lin = r["linear"]
m_star = 0.76
m2 = fit.m2_from_a0(r["a0_mas"], lin["parallax"], r["period"], m_star)
theta0 = np.array([r["period"], r["ecc"], m_star, m2, r["inc"], r["Omega"], r["omega"], lin["parallax"],
                   lin["d_ra_mas"], lin["d_dec_mas"], lin["pmra"], lin["pmdec"], r["t_peri"],
                   float(np.average(rv.rv, weights=1 / rv.err**2))])

data = E.build_data(ep, rvd, m_star_prior=(0.76, 0.05))      # rvdata=None for astrometry only
theta, mcmc = nuts.run_nuts(E.log_prob, data, theta0, num_warmup=1000, num_samples=1000)
for k_, (med, up, lo) in nuts.summarize(theta).items():
    print(f"{k_:12s} {med:10.4f} +{up:.4f} -{lo:.4f}")
mcmc.print_summary()                                           # r_hat, effective sample sizes, divergences
```

The parameter vector order is `likelihood.PARAM_NAMES`: period, ecc, m_star, m_2, inc, Omega, omega,
parallax, d_ra_mas, d_dec_mas, pmra, pmdec, t_peri, gamma. NUTS runs in a reparameterised space (log P,
sqrt(e) cos/sin omega, cos i, log masses, ...) with the Jacobian included, so the priors are flat in the
physical parameters, plus the Gaussian prior on `m_star`. One chain takes about a minute on a CPU; it is
slower on a GPU, because a single chain is latency-bound. Run candidates in parallel processes on CPU for now.

## 8. GPU benchmark notebook (Google Colab)

`notebooks/gpu_benchmark_colab.ipynb` clones the repository, builds mock sources on the DR4 pre-release
scan patterns, times the batch pipeline on CPU and GPU after JIT warm-up, scans the batch size up to
4096, checks recovery against the mock truth, and times a NUTS fit of BH3. Runtime > Change runtime
type > GPU (L4 or A100), then Runtime > Run all. The notebook runs in float32 (`PRECISION = 32` in the
first cell). Executed runs are archived in `notebooks/runs/`.

The notebook contains a read-only access token for this private repository so that it can clone the
code without any setup. Do not forward the notebook outside the collaboration.

## 9. Troubleshooting

- `Received incompatible devices for jitted computation`: you combined arrays living on different devices
  (for example CPU and GPU results). Convert with `np.asarray` first.
- Out-of-memory or a frozen machine: lower `mem_budget_gb` in `plan_chunks`, or lower `chunk`/`source_chunk`.
- Periods landing at the grid edge (e.g. 6000 d): the true period is longer than the data span and only a
  lower limit is constrained. These sources are still detected; treat the period as unconstrained.
- A bright star with chi2/dof well above 1 after refinement: refit with `fit_jitter=True`.
- `pip install -U jax` on Colab broke the CUDA plugin: do not upgrade JAX there; the notebook only installs
  numpyro and jaxopt.
