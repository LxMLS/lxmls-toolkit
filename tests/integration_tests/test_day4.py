from __future__ import division

import numpy as np
import pytest

import lxmls.parsing.dependency_parser as depp

tolerance = 1e-5
np.random.seed(4242)


@pytest.fixture(scope='module')
def dp():
    return depp.DependencyParser()


def test_exercise_1(dp):
    dp.read_data("portuguese")

    assert len(dp.reader.train_instances) == 3029
    assert len(dp.reader.test_instances) == 109
    assert len(dp.reader.word_dict) == 7621
    assert len(dp.reader.pos_dict) == 16

    assert dp.features.n_feats == 142


def test_exercise_2(dp):
    dp.train_perceptron(3)
    expected_weights = np.array([-0.33333333, 7., 5., -13.,
                                 -0.33333333, 5.33333333, 5.66666667, 4.,
                                 2., 1.66666667, 5., 1.66666667,
                                 4., 8., 4.66666667, 0.66666667,
                                 0., 3.33333333, 1.66666667, 4.33333333,
                                 -0.33333333, -2.66666667, 1., 2.33333333,
                                 -0.66666667, 2., -1.66666667, 2.33333333,
                                 -1.33333333, 1.66666667, -1.33333333, 1.,
                                 -2.66666667, -1.66666667, 3., 1.66666667,
                                 0.66666667, 0., 0.66666667, 0.66666667,
                                 1.33333333, 2.33333333, 2., -3.33333333,
                                 -5.33333333, -0.66666667, -2.33333333, 1.33333333,
                                 -2., 1., 2.66666667, 2.33333333,
                                 1.66666667, -2.33333333, -1., 1.33333333,
                                 0., 1.66666667, 1.33333333, -2.,
                                 1., 1.66666667, 0., 1.33333333,
                                 -0.33333333, -0.33333333, 0.66666667, 2.66666667,
                                 -0.33333333, 0.66666667, -2.66666667, 5.,
                                 2., -1.33333333, -2.33333333, 1.33333333,
                                 1.33333333, -2.33333333, 1., -1.33333333,
                                 5., -2., -1., 0.,
                                 2.66666667, 1.33333333, 1., 1.66666667,
                                 2., -1., 0.33333333, 0.66666667,
                                 -2.66666667, 1.33333333, -2.33333333, 1.33333333,
                                 1.66666667, 1., -1., -1.66666667,
                                 -1.33333333, 2., 2.33333333, 1.,
                                 0.33333333, 4.66666667, 4.33333333, -1.,
                                 0., -1., 1.33333333, 0.33333333,
                                 -0.66666667, -0.33333333, 0., -1.33333333,
                                 -0.66666667, 1.66666667, 2.33333333, -0.33333333,
                                 -0.33333333, 1., -4., 1.,
                                 0., 1.66666667, 2.33333333, -0.66666667,
                                 1., 1., -1.33333333, 0.66666667,
                                 -0.33333333, -0.66666667, 0.66666667, -4.,
                                 1., 1.66666667, 2., 1.,
                                 2., -0.33333333])
    assert np.allclose(dp.weights, expected_weights, rtol=tolerance)
    # FIXME: change dp.test() to allow for testing


def test_exercise_3(dp):
    dp.features.use_lexical = True
    dp.read_data("portuguese")
    assert dp.features.n_feats == 46216

    dp.train_perceptron(3)
    expected_weights = np.array([0., 6.66666667, 1., 1., 2.66666667, 1.33333333, 3.66666667, -13., 0.66666667, 3.66666667, 0.66666667,
                                 -13., 1., 1.33333333, 1., 3.33333333, -2.33333333, 8.33333333, 4.66666667, 1.33333333])
    assert np.allclose(dp.weights[:20], expected_weights, rtol=tolerance)
    assert abs(np.mean(dp.weights) - 0.79592925970803774) < tolerance
    # FIXME: change dp.test() to allow for testing

    dp.features.use_distance = True
    dp.read_data("portuguese")
    assert dp.features.n_feats == 46224
    dp.train_perceptron(3)
    expected_weights = np.array([-2., 11.66666667, 1., 1., 2., 1.66666667, -3.33333333, 4.66666667, 7.66666667,
                                 -13., 0., 7.66666667, 0., -13., 3.33333333, 2., 3., 2., 0.33333333, 2.])
    assert np.allclose(dp.weights[:20], expected_weights, rtol=tolerance)
    assert abs(np.mean(dp.weights) - 0.5340227298950041) < tolerance
    # FIXME: change dp.test() to allow for testing

    dp.features.use_contextual = True
    dp.read_data("portuguese")
    assert dp.features.n_feats == 92918
    dp.train_perceptron(3)
    expected_weights = np.array([-2., 17.33333333, 1., 1., 1., 3.66666667, -3.33333333, 5.66666667, 3.66666667,
                                 1.66666667, 1., 2., 6.66666667, 1., 3., 5.33333333, 1.66666667, 1., 1.66666667, 1.])
    assert np.allclose(dp.weights[:20], expected_weights, rtol=tolerance)
    assert abs(np.mean(dp.weights) - 0.40088393350409324) < tolerance
    # FIXME: change dp.test() to allow for testing


def test_exercise_4(dp):
    dp.train_crf_sgd(3, 0.01, 0.1)
    expected_weights = np.array([-0.0160381, 2.90118047, 0.01030002, 0.01030002, 0.0148832,
                                 0.04309881, -0.36989065, 1.64376071, 0.56773745, 0.05575176,
                                 0.04370057, 0.06350659, 0.39766793, 0.07296235, 0.13685871,
                                 0.46818032, 0.03682827, 0.04313755, 0.12355294, 0.0211694])
    assert np.allclose(dp.weights[:20], expected_weights, rtol=tolerance)
    assert abs(np.mean(dp.weights) - 0.013621573120785465) < tolerance
    # FIXME: change dp.test() to allow for testing


def test_exercise_5(dp):
    dp.read_data("english")
    assert len(dp.reader.train_instances) == 8044
    assert len(dp.reader.test_instances) == 509
    assert len(dp.reader.word_dict) == 12202
    assert len(dp.reader.pos_dict) == 48
    assert dp.features.n_feats == 338014

    dp.train_perceptron(2)
    # For 3 epochs
    # expected_weights = np.array([6.33333333, 3., 1., 1., 3., 1., 0., 8.66666667, 5.33333333, 2., 2.33333333, -1.33333333,
    #                              11.33333333, -0.33333333, -1., 8., 2.33333333, 2.66666667, 0., 1.])
    # For 2 epochs
    expected_weights = np.array([6., 3., 1., 1., 2.5, 1., 1., 8., 5., 2., 2.5, -1.5, 11.5, 0., -1., 8., 2., 2.5, 0., 0.5])
    assert np.allclose(dp.weights[:20], expected_weights, rtol=tolerance)
    # assert abs(np.mean(dp.weights) - 0.4546261397456911) < tolerance  # for 3 epochs
    assert abs(np.mean(dp.weights) - 0.410049583745) < tolerance  # for 2 epochs
    # FIXME: change dp.test() to allow for testing


def test_exercise_6():
    dep_parser = depp.DependencyParser()
    dep_parser.features.use_lexical = True
    dep_parser.features.use_distance = True
    dep_parser.features.use_contextual = True
    dep_parser.read_data("english")
    assert len(dep_parser.reader.train_instances) == 8044
    assert len(dep_parser.reader.test_instances) == 509
    assert len(dep_parser.reader.word_dict) == 12202
    assert len(dep_parser.reader.pos_dict) == 48
    assert dep_parser.features.n_feats == 338014

    dep_parser.projective = True

    dep_parser.train_perceptron(2)
    # For 3 epochs
    # expected_weights = np.array([9.33333333, 2.33333333, 1., 1., 3.33333333, 0., 0., 3.33333333, 6.66666667, 3.33333333, 1.66666667, 0.,
    #                              10.66666667, 1., -1.66666667, 9.66666667, 2., 3.33333333, 0.33333333, 1.])
    # For 2 epochs
    expected_weights = np.array([8.5, 1.5, 1., 1., 3.5, 0., 0.5, 3.5, 6.5, 3.5, 1.5, 0., 10.5, 0.5, -1., 10., 2.5, 3., 0.5, 1.5])
    assert np.allclose(dep_parser.weights[:20], expected_weights, rtol=tolerance)
    # assert abs(np.mean(dep_parser.weights) - 0.43731324738028604) < tolerance  # for 3 epochs
    assert abs(np.mean(dep_parser.weights) - 0.392037016218) < tolerance  # for 2 epochs
    # FIXME: change dp.test() to allow for testing


if __name__ == '__main__':
    pytest.main([__file__])
