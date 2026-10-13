# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`darkcomp` is a JAX code for weighing dark companions (neutron stars, black holes, white dwarfs)
to luminous stars from Gaia astrometry plus radial velocities. Two data regimes are supported:

1. **DR3 summary solutions** (El-Badry et al. 2024, arXiv:2405.00089, Sect. 5): Gaia publishes a
   12/15-parameter orbital solution with covariance; `likelihood.py` forward-models it from physical
   parameters and adds an RV chi-square.
2. **Epoch (along-scan) astrometry** as in DR4 (Gaia Collaboration/Panuzzo et al. 2024 for BH3;
   El-Badry et al. 2024 arXiv:2411.00088 for the fitting cascade): `epoch.py` models each 1-D CCD
   measurement; `batch.py` is the many-source, GPU-shaped search.

Design rule: every likelihood is jit/vmap/grad-able with static shapes and no Python control flow
on traced values. **The dev machine has no usable GPU and limited RAM**: never `vmap` more than a
few hundred (n_obs, 9) design matrices at once (a 480k-point vmap once froze the machine); grid
searches go through chunked `lax.map` (`epoch.profile_loglike_batch`, `batch.batch_period_search`).
Benchmarks on a GPU run in Google Colab via `notebooks/gpu_benchmark_colab.ipynb`. GitHub:
`git@github.com:Hanyuan0908/astrometry_jax.git` (private).

## Environment and commands

No `uv`/venv; use the conda env `astro312` (python 3.12, jax 0.7 CPU, numpyro, astroquery, pandas, pytest):

```bash
PY=/Users/hanyuan/miniforge3/envs/astro312/bin/python
$PY -m pytest -q                              # full suite (~10 s)
$PY -m pytest -q tests/test_likelihood.py -k paper   # one test / pattern
$PY scripts/fit_source.py J0553-1349          # full paper-settings fit (64 walkers, 3000+3000 steps)
$PY scripts/fit_source.py J0553-1349 --steps 300 --burn 300 --no-rvs   # quick, Gaia-only
```

`$PY -m pip install -e .` is not required; tests and scripts put the repo root on `sys.path`.
Network access (astroquery to the Gaia archive) is only needed for sources not in
`data/gaia_dr3_nss_fixture.json`; tests never hit the network.

## Architecture (read in this order)

### Epoch astrometry (DR4 path)
- `darkcomp/epoch.py` — abscissa model `w = (dra + pmra t + BX + GY) sin psi + (ddec + pmdec t + AX + FY) cos psi + plx f`.
  At fixed (P, e, phase) it is linear in 9 parameters, so `profile_loglike` is one weighted
  least-squares solve; `period_search` evaluates a (P, e, phase) grid in memory-safe chunks and
  `refine_profile` polishes with L-BFGS on the JAX gradient (optionally fitting a jitter term, needed
  for bright stars like HD 114762). `predict_abscissa`/`log_prob` give the full 14-parameter
  likelihood for joint fits with RVs. Reference epoch J2017.5 (`DR4_REF_JD`); `build_data` re-references RV times.
- `darkcomp/dr4.py` — parser for the DR4 `epoch_astrometry` VOTable (per-CCD arrays exploded;
  keep `used_by_agis_al` and not `multipeak`, as ESA's gaia-supdate does). Time = ns since 2010.0 TCB + bary corr.
- `darkcomp/batch.py` — stacked/padded multi-source pipeline: vmapped single-star fit, `lax.map`
  over sources of the chunked periodogram, bounded `jaxopt.LBFGSB` refinement of the top-3 peaks,
  Campbell elements, significance and companion mass. Grid e <= 0.6 by design: high-e grid points
  overfit noise (spurious e -> 1 solutions).
- `darkcomp/nuts.py` — numpyro NUTS on a reparameterised vector (log P, sqrt(e) cos/sin omega, cos i, ...)
  with the Jacobian added so priors stay flat in physical parameters; dense mass matrix; start at the MAP.
- `darkcomp/mock.py` — mock sources reusing real scan patterns (not a gaiamock replacement).
- `data/gaia_bh3/` (ESA, CDS J/A+A/686/L2) and `data/dr4_prerelease/` (ESA, 12 sources: 9 singles/QSOs
  + HD 114762, Gaia-4, Gaia BH3) are the real-data benchmarks.

### DR3 summary solutions

- `darkcomp/orbit.py` — all orbital geometry and the conventions. Thiele-Innes A,B,F,G (mas) and
  C,H (au) exactly as in the Gaia archive / paper eqs. 5-10; `campbell_from_thiele_innes` inverts
  them with cancellation-free identities. RV curve `gamma + K[cos(w+f) + e cos w]`. Units are
  au / day / Msun / mas / km s^-1 throughout (`constants.py`); angles in radians inside code.
- `darkcomp/kepler.py` — fixed-iteration Newton solver with analytic `custom_jvp` (implicit
  differentiation), so gradients don't unroll the loop. Do not add data-dependent `while` loops here.
- `darkcomp/gaia.py` — `GaiaSolution`: mean vector + covariance rebuilt from `corr_vec` (strict
  lower triangle, row-major). Parameter order is the archive order and *is* the order of the
  predicted vector in the likelihood. ra/dec are replaced by `d_ra_mas`, `d_dec_mas` offsets (0 at
  the archive position) because `ra_error`/`dec_error` are in mas.
- `darkcomp/likelihood.py` — the 14 free parameters (`PARAM_NAMES`; optional 15th `eps` flux
  ratio), `derived()` (a, a1, a0, K, A..H, mass functions), `gaia_vector()`, `log_prob(theta, data)`.
  `data` is an array-only pytree from `build_data()`: solution type is inferred from
  `gaia_mean.shape[0]` (12 Orbital / 15 AstroSpectroSB1) so it is static under jit; the RV term is
  gated by the scalar `rv_weight` (set 0 for the paper's "Gaia-only" fits). Priors are flat boxes
  plus a Gaussian on `m_star` from the SED mass.
- `darkcomp/sampler.py` — Goodman-Weare stretch move in pure JAX (`lax.scan` over jitted
  red/blue half-ensemble updates). `run_ensemble(key, walkers, data, log_prob, n_steps)` is jitted
  with `log_prob` static, so pass the same function object each call to avoid recompiles.
- `darkcomp/fit.py` — `initial_guesses()` converts the Gaia solution to Campbell elements and
  solves for M2 (replacing `nsstools`); returns both astrometric branches `(w, W)` and
  `(w+pi, W+pi)` (equivalent to i -> -i). `fit_source()` L-BFGS-optimises each branch on the JAX
  gradient, keeps the higher-likelihood one (paper Section 5.1), then samples.
- `data/` — paper fixtures: all 303 published RVs (`elbadry2024_rvs.csv`), SED masses with the
  paper's 0.03 Msun floor (`elbadry2024_star_params.csv`), and cached Gaia rows for J0553-1349 and
  J2057-4742. Times are HJD; internally everything is days after JD 2457389.0 (Gaia J2016.0).

## Conventions that are easy to get wrong

- Branch degeneracy: astrometry only sees cos i, so (w, W) and (w+pi, W+pi) give identical
  A,B,F,G but flip the RV curve sign. `tests/test_likelihood.py::test_rv_sign_convention_gives_paper_chi2`
  pins the convention against the published J0553-1349 solution; keep it passing.
- `inc` is sampled in [0, pi]; the paper reports i > 90 deg for retrograde orbits, so do not fold.
- `campbell_from_thiele_innes` returns W in [0, pi); that is only a starting point, the sampler
  is free to wander. Wrap angles with `mod 2pi` only when summarising.
- Gaia `t_periastron` is relative to JD 2457389.0 and is sampled directly; it is periodic in P
  and may drift by whole periods in long chains.
- No parallax zero-point correction is applied (the paper did not either).

## Validation reference

- `tests/test_likelihood.py`: J0553-1349 (DR3 2995961897685517312) from arXiv:2405.00089 Appendix C:
  a = 0.851 au, a0 = 1.229 mas, K = 24.506 km/s, M2 = 1.33 Msun, chi2/N_RV = 1.2.
- `tests/test_epoch.py`: Gaia BH3 from its CDS epoch data, astrometry-only: P = 4194.7 +- 112 d,
  a0 = 27.07 mas, i = 110.66 deg, f(M) = 32.03 Msun. NUTS joint fit gives M_BH = 33.1 +- 0.5 (paper 32.70 +- 0.82).
- `tests/test_batch.py`: the 9 DR4 pre-release single stars/QSOs must stay null (significance < 6),
  HD 114762 and BH3 detected, mock orbits recovered. When touching geometry or units, run these first.
