import sys
import os
import pytest

LXMLS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, LXMLS_ROOT)

from numpy import array, array_equal, allclose
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lxmls.readers import galton

tolerance = 1e-5

@pytest.fixture(scope='module')
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
    expected_n = [array([  0.,  14.,  23.,  66., 289., 219., 183.,  68.,  43.,  23.]), array([ 12.,  32., 107., 117., 138., 120., 167., 163.,  41.,  31.])]
    expected_bins = array([61.7, 62.9, 64.1, 65.3, 66.5, 67.7, 68.9, 70.1, 71.3, 72.5, 73.7])
    assert allclose(n, expected_n, tolerance)
    assert allclose(bins, expected_bins, tolerance)

if __name__ == '__main__':
    pytest.main([__file__])
