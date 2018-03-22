import sys
import os
import pytest

LXMLS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, LXMLS_ROOT)
import numpy as np

import lxmls.parsing.dependency_parser as depp

tolerance = 1e-5

@pytest.fixture
def dp():
    return depp.DependencyParser()


@pytest.fixture
def dp_en(dp):
    dp.features.use_lexical = True
    dp.features.use_distance = True
    dp.features.use_contextual = True
    dp.read_data("english")
    assert len(dp.reader.train_instances) == 8044
    assert len(dp.reader.test_instances) == 509
    assert len(dp.reader.word_dict) == 12202
    assert len(dp.reader.pos_dict) == 48
    assert dp.features.n_feats == 338014
    return dp


def test_exercise_1(dp):
    dp.read_data("portuguese")

    assert len(dp.reader.train_instances) == 3029
    assert len(dp.reader.test_instances) == 109
    assert len(dp.reader.word_dict) == 7621
    assert len(dp.reader.pos_dict) == 16

    assert dp.features.n_feats == 142


def test_exercise_2(dp):
    dp.read_data("portuguese")
    dp.train_perceptron(1)
    expected_weights = np.array([   0.,   8.,   5., -13.,   0.,   5.,   7.,   3.,   3.,   2.,   7.,
                                    2.,   4.,   9.,   5.,   0.,   1.,   4.,   2.,   5.,   0.,  -3.,
                                    1.,   2.,  -1.,   0.,  -1.,   2.,   0.,   1.,  -2.,   1.,  -4.,
                                    -2.,   2.,  -1.,   0.,   0.,   1.,   0.,   2.,   1.,   3.,  -2.,
                                    -6.,   0.,  -3.,   2.,  -1.,   1.,   2.,   1.,   2.,  -2.,   0.,
                                    1.,  -1.,   2.,  -1.,  -4.,  -1.,   2.,   0.,   0.,   0.,  -2.,
                                    0.,   2.,   0.,  -1.,  -3.,   3.,   1.,  -1.,  -3.,   0.,   2.,
                                    0.,   2.,  -2.,   4.,  -2.,   0.,   1.,   3.,   2.,   0.,   0.,
                                    3.,   0.,   0.,   0.,  -3.,   2.,  -2.,   2.,   2.,   0.,   0.,
                                    -1.,  -1.,   1.,   0.,   0.,  -1.,   4.,   3.,  -1.,   2.,  -1.,
                                    1.,   1.,  -1.,  -2.,  -1.,  -1.,  -1.,   2.,   2.,   1.,   0.,
                                    1.,  -2.,   1.,   0.,   1.,   2.,   0.,   2.,   1.,  -6.,   0.,
                                    -1.,   0.,   1.,  -3.,  -2.,   0.,   1.,   1.,   1.,   0.])


    print(expected_weights)
    print(dp.weights)
    assert np.allclose(dp.weights, expected_weights, rtol=tolerance)
    # FIXME: change dp.test() to allow for testing


def test_exercise_3a(dp):
    dp.features.use_lexical = True
    dp.read_data("portuguese")
    assert dp.features.n_feats == 46216

    dp.train_perceptron(1)
    expected_weights = np.array([ -1.,   8.,   1.,   1.,   2.,   2.,   3., -13.,   1.,   3.,   1.,
                                -13.,   2.,   1.,   0.,   4.,  -6.,   6.,   5.,   1.])
    assert np.allclose(dp.weights[:20], expected_weights, rtol=tolerance)
    assert abs(np.mean(dp.weights) - 0.4787519473775316) < tolerance
    # FIXME: change dp.test() to allow for testing


def test_exercise_3b(dp):
    dp.features.use_lexical = True
    dp.features.use_distance = True
    dp.read_data("portuguese")
    assert dp.features.n_feats == 46224
    dp.train_perceptron(1)
    expected_weights = np.array([ -1.,  12.,   1.,   1.,   2.,   2.,  -4.,   5.,   9., -13.,   0.,
                                   9.,   0., -13.,   4.,   1.,   3.,   1.,   0.,   3.])
    assert np.allclose(dp.weights[:20], expected_weights, rtol=tolerance)
    assert abs(np.mean(dp.weights) - 0.3360375562478366) < tolerance
    # FIXME: change dp.test() to allow for testing


def test_exercise_3c(dp):
    dp.features.use_lexical = True
    dp.features.use_distance = True
    dp.features.use_contextual = True
    dp.read_data("portuguese")
    assert dp.features.n_feats == 92918
    dp.train_perceptron(1)
    expected_weights = np.array([-3., 13.,  1.,  1.,  1.,  4., -4.,  8.,  4.,  1.,  1.,  1.,  5.,
                                  1.,  3.,  4.,  1.,  1.,  1.,  1.])

    assert np.allclose(dp.weights[:20], expected_weights, rtol=tolerance)
    assert abs(np.mean(dp.weights) - 0.29635807916657697) < tolerance
    # FIXME: change dp.test() to allow for testing


def test_exercise_4(dp):
    dp.features.use_lexical = True
    dp.features.use_distance = True
    dp.features.use_contextual = True
    dp.read_data("portuguese")

    dp.train_crf_sgd(1, 0.01, 0.1)
    expected_weights = np.array([ 0.02128726,  2.85216804,  0.02317114,  0.02317114,  0.02581292,
                                  0.07219983, -0.37288885,  1.60089409,  0.5567024 ,  0.06519791,
                                  0.05362943,  0.07874481,  0.40610712,  0.07791399,  0.14396417,
                                  0.46424806,  0.04734969,  0.05299581,  0.12391761,  0.03160131])

    assert np.allclose(dp.weights[:20], expected_weights, rtol=tolerance)
    assert abs(np.mean(dp.weights) - 0.01263421546824582) < tolerance
    # FIXME: change dp.test() to allow for testing


def test_exercise_5(dp_en):
    dp_en.train_perceptron(1)
    expected_weights = np.array([ 9.,  1.,  1.,  1.,  2.,  1.,  2.,  7.,  6.,  3.,  2.,  2., 11., 0.,  2., 10.,  2.,  3.,  1.,  1.])

    assert np.allclose(dp_en.weights[:20], expected_weights, rtol=tolerance)
    assert abs(np.mean(dp_en.weights) - 0.3450330459685102) < tolerance
    # FIXME: change dp.test() to allow for testing


def test_exercise_6(dp_en):
    dp_en.projective = True

    dp_en.train_perceptron(1)
    expected_weights = np.array([ 8.,  0.,  1.,  1.,  4.,  0.,  0.,  4.,  5.,  4.,  2.,  1., 12., 2.,  1., 12.,  2.,  3.,  2.,  1.])
    assert np.allclose(dp_en.weights[:20], expected_weights, rtol=tolerance)
    assert abs(np.mean(dp_en.weights) - 0.3313708899631376) < tolerance
    # FIXME: change dp.test() to allow for testing


if __name__ == '__main__':
    pytest.main([__file__])
