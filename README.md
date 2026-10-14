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
- Mocks on real scan patterns (`mock.py`) are kept for benchmarks; the physically meaningful mocks now come from
  gaiamock through `gaiamock_bridge.py`, which requires a 1 GB local gaiamock install (see docs/RUNNING.md).

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

**gaiamock cross-check**: mock epoch astrometry is generated by gaiamock (El-Badry et al. 2024, the community
simulator: nominal scanning law at any sky position, G-dependent along-scan errors, DR4 time span) through
`darkcomp/gaiamock_bridge.py`. On identical gaiamock data, gaiamock's own `fit_full_astrometric_cascade`
and our pipeline return the same P, e, a0, parallax and inclination to the printed precision (4 test
systems, P 150 to 1500 d, G 12 to 16), and `tests/test_batch.py` recovers P to 2%, a0 to 10% and M2 to 15%.

**gaiamock against the real DR4 pre-release data** (`scripts/benchmark_gaiamock.py`, 12 sources, G 7 to 20):
gaiamock predicts the same number of DR4 transits as observed to within 0 to 25% (e.g. 63 vs 63 for BH3,
64 vs 63 for HD 114762) and 90 to 100% of the real transits fall within 12 h of a predicted one (the
0.9 deg healpix cell can be swept one revolution or one field of view earlier or later, so only 35 to
90% coincide to the minute). Scan angles agree to 0.1 to 0.7 deg rms and parallax factors to 0.003.
Its per-CCD error model is within 5 to 10% of the reported errors for G 14 to 20 and 50% larger than
the reported errors at G 7 to 12, but the reported bright-star errors underestimate the actual scatter:
the real BH3 fit has chi2/dof 3.05, i.e. an effective per-CCD error of 0.149 mas against gaiamock's
0.133. Single-star chi2/N is 0.95 to 1.08 on the real nulls and 0.91 to 1.05 on gaiamock realisations.
Injecting the fitted BH3 orbit into 20 gaiamock realisations at BH3's position and magnitude returns
P 4242 +- 180 d, e 0.730 +- 0.009, a0 27.4 +- 0.9 mas, i 110.5 +- 0.2 deg, parallax +- 0.016 mas and
significance 1020 +- 43, against the real data's a0 27.11 mas, i 110.5 deg, parallax +- 0.011 mas and
significance 1099 once its excess noise is included. gaiamock is therefore a faithful, slightly
conservative, stand-in for DR4 epoch astrometry.

**Recovery on 128 gaiamock systems** (M2 log-uniform 0.2 to 10 Msun, distance 200 pc to 3 kpc, solar-type
primary so G 11 to 17, P 30 to 1500 d, e up to 0.6, random sky): 112 of 128 detected at 6 sigma, 110 of
those with the period within 5%; the non-detections have a0 of 0.08 mas median, 0.21 mas at most. Over the
110 recovered orbits the median relative errors are 1.4% in a0, 0.7% in parallax, 3% in M2, 1.5 deg in
inclination; eccentricity is the least constrained (median 12%). float32 and float64 give the identical
detection set, 111 of 112 periods equal to 0.1%, a0 and parallax differing by 1e-5 mas in the median.

**Detectability versus companion mass and distance** (`scripts/detectability.py`, gaiamock mocks at random
sky positions, DR4 scanning law, solar-type primary with M_G = 4.5 so G = 13.0, 14.5, 16.0, 18.0, 19.5 at
0.5, 1, 2, 5, 10 kpc; dark companion; P in 30 to 1500 d; 20 systems per cell; detection = 5 sigma):

| M2 [Msun] | 500 pc | 1 kpc | 2 kpc | 5 kpc | 10 kpc |
|---|---|---|---|---|---|
| 0.1 | 1.00 | 0.70 | 0.05 | 0.00 | 0.05 |
| 0.3 | 1.00 | 0.95 | 0.45 | 0.10 | 0.00 |
| 0.6 | 1.00 | 1.00 | 0.65 | 0.05 | 0.05 |
| 1.0 | 1.00 | 1.00 | 0.95 | 0.30 | 0.10 |
| 1.4 | 1.00 | 1.00 | 1.00 | 0.20 | 0.10 |
| 3.0 | 1.00 | 1.00 | 1.00 | 0.30 | 0.05 |
| 10 | 1.00 | 1.00 | 1.00 | 0.35 | 0.10 |

A 1.4 Msun neutron star around a solar-type star is recovered in every case out to 2 kpc (G 16, a0 about
0.3 mas); beyond that the primary is fainter than G 18 and the per-transit error (0.4 mas and more) kills
the signal. The recovered median M2 is within 1% of truth at 0.5 and 1 kpc and within 20% at 2 kpc.
Null test on 200 gaiamock single stars: significance exceeds 5 in 8% of cases (25% at G 13, where gaiamock
adds unmodelled bright-star noise), exceeds 6 in 0.5%, never exceeds 7. Use 6 sigma, or the jitter-aware
refinement for bright stars, when a clean candidate list matters. The template-based variants (real
pre-release scan patterns, errors as measured or scaled 5x) are in `docs/figures/detectability_bright.*`
and `detectability_noise5.*`; figure for gaiamock: `docs/figures/detectability_gaiamock.png`.

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
scripts/          fit_source.py (DR3 paper fit), detectability.py (mass-distance grid, gaiamock or templates), precision_check.py
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
- Mock epoch astrometry: gaiamock, El-Badry et al. 2024, arXiv:2411.00088, https://github.com/kareemelbadry/gaiamock.

## References

- El-Badry et al. 2024, "A population of neutron star candidates in wide orbits from Gaia astrometry", arXiv:2405.00089
- El-Badry et al. 2024, arXiv:2411.00088 (epoch-astrometry fitting cascade)
- Gaia Collaboration, Panuzzo et al. 2024, Gaia BH3, A&A 686, L2
