import numpy as np
import jax.numpy as jnp
from darkcomp import orbit
from darkcomp.gaia import corr_matrix_from_vec
from tests.conftest import F32

TOL = 1e-5 if F32 else 1e-9


def test_thiele_innes_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(200):
        a0 = rng.uniform(0.1, 5); w = rng.uniform(0, 2 * np.pi); W = rng.uniform(0, 2 * np.pi); i = rng.uniform(0.01, np.pi - 0.01)
        A, B, F, G = (float(x) for x in orbit.thiele_innes_ABFG(a0, w, W, i))
        a0r, wr, Wr, ir = orbit.campbell_from_thiele_innes(A, B, F, G)
        A2, B2, F2, G2 = (float(x) for x in orbit.thiele_innes_ABFG(a0r, wr, Wr, ir))
        assert np.allclose([A, B, F, G], [A2, B2, F2, G2], atol=TOL)
        assert abs(a0r - a0) < TOL and abs(ir - i) < TOL
        assert 0 <= Wr < np.pi


def test_corr_vec_ordering_lower_triangle_rowwise():
    n = 4
    vec = np.arange(1, n * (n - 1) // 2 + 1, dtype=float) / 10
    C = corr_matrix_from_vec(vec, n)
    assert C[1, 0] == 0.1 and C[2, 0] == 0.2 and C[2, 1] == 0.3 and C[3, 0] == 0.4 and C[3, 2] == 0.6
    assert np.allclose(C, C.T) and np.allclose(np.diag(C), 1)


def test_kepler_third_law_earth():
    a = float(orbit.semimajor_axis_au(365.25636, 1.0, 3.0e-6))
    assert abs(a - 1.0) < 2e-4
