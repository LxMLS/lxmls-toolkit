import matplotlib
import pytest
from numpy import allclose, array

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lxmls.readers import galton

tolerance = 1e-5


@pytest.fixture(scope="module")
def galton_data():
    return galton.load()


def test_galton_data(galton_data):
    mean = galton_data.mean(0)
    expected_mean = array([68.30818966, 68.08846983])
    assert allclose(mean, expected_mean, tolerance)

    std = galton_data.std(0)
    expected_std = array([1.78637014, 2.51658435])
    assert allclose(std, expected_std, tolerance)

    n, bins, _ = plt.hist(galton_data)
    expected_n = [
        array([0.0, 14.0, 23.0, 66.0, 289.0, 219.0, 183.0, 68.0, 43.0, 23.0]),
        array([12.0, 32.0, 107.0, 117.0, 138.0, 120.0, 167.0, 163.0, 41.0, 31.0]),
    ]
    expected_bins = array([61.7, 62.9, 64.1, 65.3, 66.5, 67.7, 68.9, 70.1, 71.3, 72.5, 73.7])
    assert allclose(n, expected_n, tolerance)
    assert allclose(bins, expected_bins, tolerance)


if __name__ == "__main__":
    pytest.main([__file__])
