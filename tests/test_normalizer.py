from nanoppo.normalizer import Normalizer
import numpy as np

def test_normalizer_1d():
    # Tolerance for floating point comparisons
    tol = 1e-5

    # Test for 1D
    normalizer_1d = Normalizer(dim=1)
    data_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
    for d in data_1d:
        normalizer_1d.observe(d)
    normalized_1d = normalizer_1d.normalize(data_1d)
    assert abs(np.mean(normalized_1d)) < tol
    assert abs(np.std(normalized_1d) - 1.0) < tol

def test_normalizer_2d():
    # Tolerance for floating point comparisons
    tol = 1e-5

    # Test for 2D
    normalizer_2d = Normalizer(dim=2)
    data_2d = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
    for d in data_2d:
        normalizer_2d.observe(d)
    normalized_2d = normalizer_2d.normalize(data_2d)
    assert np.all(abs(np.mean(normalized_2d, axis=0)) < tol)
    assert np.all(abs(np.std(normalized_2d, axis=0) - 1.0) < tol)

def test_normalizer_3d():
    # Tolerance for floating point comparisons
    tol = 1e-5

    # Test for 3D
    normalizer_3d = Normalizer(dim=3)
    data_3d = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0], [5.0, 6.0, 7.0]])
    for d in data_3d:
        normalizer_3d.observe(d)
    normalized_3d = normalizer_3d.normalize(data_3d)
    assert np.all(abs(np.mean(normalized_3d, axis=0)) < tol)
    assert np.all(abs(np.std(normalized_3d, axis=0) - 1.0) < tol)
