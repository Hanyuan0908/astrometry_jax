# darkcomp: weighing dark companions with Gaia astrometry, in JAX

`darkcomp` finds and characterises luminous stars orbited by dark companions (neutron stars, black holes,
white dwarfs) from Gaia astrometry, optionally combined with radial velocities. Everything is written in
JAX so that the same code runs on a laptop CPU and, unchanged, on a GPU where thousands of sources are
searched in one batched call.

Two data regimes are covered:

| regime | input | what the code does | module |
|---|---|---|---|
| **Gaia DR3 summary solutions** | the 12/15-parameter `nss_two_body_orbit` row with its covariance, plus RVs | forward-models the archive solution from physical parameters (P, e, M*, M2, i, Omega, omega, parallax, ...) and samples the posterior; reproduces El-Badry et al. 2024 (arXiv:2405.00089) Sect. 5 | `likelihood.py`, `fit.py`, `sampler.py` |
| **Epoch (along-scan) astrometry**, i.e. Gaia DR4 | per-CCD 1-D abscissae with scan angles and parallax factors | single-star fit, profile-likelihood periodogram over (P, e, phase), Levenberg-Marquardt refinement, Campbell elements, detection significance and companion mass, for many sources at once; NUTS posterior for selected candidates | `epoch.py`, `batch.py`, `dr4.py`, `nuts.py` |

See [`docs/RUNNING.md`](docs/RUNNING.md) for installation and step-by-step usage, and
[`CLAUDE.md`](CLAUDE.md) for the internal architecture and conventions.

## Status (13 October 2026)

Working and validated:

- DR3 joint astrometry + RV fits reproduce the published neutron-star candidate solutions.
- Epoch-astrometry search reproduces Gaia BH3 and the DR4 pre-release orbits, and gives no false
  detections on the DR4 pre-release single stars and quasars.
- Batched multi-source search runs on GPU at >100 sources per second (A100), with identical results to the CPU.
- One global precision switch (`DARKCOMP_PRECISION=32|64`); float32 gives the same detections and
  parameters as float64 to ~1e-5 relative. TF32 tensor-core matmuls are disabled so float32 means float32.

Not yet done:

- NUTS posteriors are per source and latency-bound on GPU (one chain: 67 s CPU, 127 to 162 s GPU). A
  vectorised many-candidate NUTS (numpyro `chain_method="vectorized"` or blackjax under `vmap`) is the next step.
- Not run on a full DR4 catalogue yet (DR4 is scheduled for December 2026). The search has been exercised on
  the 12 pre-release sources and on mock populations built on their real scan patterns.
- The mock generator reuses real Gaia scan patterns and per-CCD errors; it is not a replacement for gaiamock.

## Validation: tests performed

All of the following are in `tests/` and run under both precisions (`pytest -q`, 20 tests, about 30 s on a laptop).

**DR3 path, against El-Badry et al. 2024 tables**

| source | quantity | this code | paper |
|---|---|---|---|
| J0553-1349 (Gaia DR3 2995961897685517312) | M2 [Msun] | 1.326 +- 0.044 | 1.33 +- 0.05 |
| J0553-1349 | a0 [mas], K [km/s] | 1.229, 24.5 | 1.229, 24.506 |
| J0553-1349 | chi2 / N_RV | 1.2 | 1.2 |
| J2057-4742 (Gaia DR3 6481502062263141504) | inclination [deg] | 130.2 | 130.1 +- 1.1 |

Both astrometric branches (omega, Omega) and (omega + pi, Omega + pi) are tried and the RV sign convention
is pinned by a test against the published solution. A full paper-settings fit (64 walkers, 3000 + 3000
steps) takes about 2 s.

**Epoch astrometry, Gaia BH3 (Panuzzo et al. 2024), astrometry only, from the CDS epoch table**

| quantity | this code | paper |
|---|---|---|
| P [d] | 4236 | 4194.7 +- 112 |
| e | 0.728 | 0.7291 +- 0.0048 |
| a0 [mas] | 27.30 | 27.07 +- 0.56 |
| i [deg] | 110.59 | 110.66 +- 0.16 |
| parallax [mas] | 1.681 | 1.6747 +- 0.0094 |
| f(M) [Msun] | 31.9 | 32.03 |
| M_BH with M* = 0.76 [Msun] | 33.3 | 32.70 +- 0.82 |

The reduced chi2 is 0.92. The coarse periodogram peaks at an alias near 2226 d because the data span
(1861 d) is shorter than the period; the LM refinement of the top peaks recovers the true solution.

**Joint astrometry + RVS NUTS fit of BH3**: P = 4217 +- 85 d, a0 = 27.21 +- 0.43 mas, i = 110.61 +- 0.09 deg,
gamma = -357.23 +- 0.41 km/s, M_BH = 33.1 +- 0.5 Msun (paper combined fit 32.70 +- 0.82); effective sample
size about 1000 of 1000, no divergences, 26 s on CPU.

**Gaia DR4 pre-release epoch astrometry (12 sources released by ESA in June 2026)**

| sources | result |
|---|---|
| 9 single stars and quasars | null: single-star chi2/N 0.94 to 1.07, best-orbit Delta chi2 10 to 22, significance 3 to 5 sigma |
| HD 114762 | detected, P = 83.8 d (231 sigma from the batch pipeline; 76 sigma once a 0.35 mas jitter term brings chi2/dof to 1.02) |
| Gaia-4 | detected at 42 sigma, P = 579 d, a0 = 0.26 mas |
| Gaia BH3 | detected at 1080 sigma |

**Mock recovery (batch pipeline)**: 128 mock orbits on the real scan patterns of the 8 best-measured
pre-release sources, P log-uniform in 30 to 2500 d, a0 in 0.05 to 5 mas, e up to 0.8. 126 of 128 detected
(the two misses have a0 = 0.06 mas), 104 periods within 5% of truth; the remaining detections have either
P longer than the data span (aliases) or a0 below 0.12 mas. CPU float32, CPU float64, L4 and A100 all give
exactly these numbers.

**Precision**: float32 versus float64 on 128 mock sources: identical detection set, 122 of 126 periods agree
to better than 0.1% (the rest to 0.35%, all with P longer than the data span), a0 differences of order
1e-5 mas, chi2 relative difference 1e-6. The DR3 MAP agrees to five digits.

## Benchmarks achieved

Batched search, 12,000 trial orbits per source (300 periods x 5 eccentricities x 8 phases), top-3 peak
refinement, float32, timed after JIT warm-up (median of repeated runs; executed notebooks are in `notebooks/runs/`).

| device | batch size N | sources x trials per step | time | throughput |
|---|---|---|---|---|
| Apple M5 Pro (CPU, float64) | 64 | 32 x 256 | 15.2 s | 4.2 sources/s |
| Apple M5 Pro (CPU, float32) | 64 | 32 x 256 | 10.5 s | 6.1 sources/s |
| Colab CPU | 128 | 32 x 256 | 75 s | 1.7 sources/s |
| NVIDIA L4 (23 GB) | 128 | 128 x 256 | 5.2 s | 24.5 sources/s |
| NVIDIA L4 | 4096 | 350 x 256 | 92.5 s | 44.3 sources/s |
| NVIDIA A100-80GB | 128 | 128 x 256 | 3.7 s | 34.9 sources/s |
| NVIDIA A100-80GB | 1024 | 1024 x 256 | 8.3 s | 123.7 sources/s |
| NVIDIA A100-80GB | 4096 | 1260 x 256 | 33.1 s | 123.8 sources/s |

The A100 is compute-bound from about 1000 sources per step; at 124 sources/s one A100 searches a million
sources in about 2.2 hours. The per-step batch is planned from the device memory by `batch.plan_chunks`, so
the same notebook adapts to any GPU.

Single-candidate NUTS (BH3, astrometry + RVs, 800 warm-up + 1000 samples): 67 s on CPU, 127 s on L4,
162 s on A100. A single chain cannot use a GPU; see "Not yet done" above.

## Repository layout

```
darkcomp/         the package (see CLAUDE.md for a module-by-module description)
scripts/          fit_source.py (DR3 paper fit), precision_check.py
notebooks/        gpu_benchmark_colab.ipynb and archived executed runs in notebooks/runs/
tests/            pytest suite (20 tests)
data/             El-Badry et al. 2024 RVs and SED masses, cached DR3 rows, Gaia BH3 epoch data (CDS),
                  Gaia DR4 pre-release epoch astrometry (ESA)
docs/RUNNING.md   how to install and run
```

## Data credits

- Gaia BH3 epoch astrometry and RVS velocities: Gaia Collaboration, Panuzzo et al. 2024, CDS J/A+A/686/L2
  (licence in `data/gaia_bh3/LICENSE.txt`).
- Gaia DR4 pre-release epoch astrometry: ESA, https://www.cosmos.esa.int/web/gaia/dr4-prerelease.
- Radial velocities and stellar parameters of the neutron-star candidates: El-Badry et al. 2024, arXiv:2405.00089.

## References

- El-Badry et al. 2024, "A population of neutron star candidates in wide orbits from Gaia astrometry", arXiv:2405.00089
- El-Badry et al. 2024, arXiv:2411.00088 (epoch-astrometry fitting cascade)
- Gaia Collaboration, Panuzzo et al. 2024, Gaia BH3, A&A 686, L2
