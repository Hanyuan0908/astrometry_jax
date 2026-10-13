# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`darkcomp` is a JAX implementation of the joint Gaia-astrometry + radial-velocity orbit fit used to
weigh dark companions (neutron stars, black holes, white dwarfs) to luminous stars, following
El-Badry et al. 2024, "A population of neutron star candidates in wide orbits from Gaia astrometry"
(arXiv:2405.00089, Section 5 and Appendices). The paper's method is: Gaia DR3 publishes a 12- or
15-parameter orbital solution with a full covariance; we forward-model those parameters from
physical Campbell elements plus masses, add a chi-square over follow-up RVs, and sample the posterior.
The design goal is that every piece of the likelihood is jit/vmap/grad-able so it runs on a GPU and
can be batched over many sources. There is no GPU on the dev machine (Apple M5, CPU-only JAX); do
not assume one, but keep everything shape-static and free of Python-side control flow on traced values.

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

`tests/test_likelihood.py` reproduces J0553-1349 (Gaia DR3 2995961897685517312) from the paper's
Appendix C: a = 0.851 au, a0 = 1.229 mas, K = 24.506 km/s, M2 = 1.33 Msun, chi2/N_RV = 1.2.
When touching geometry or units, run that file first.
