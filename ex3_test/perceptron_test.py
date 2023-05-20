import dataclasses
import unittest

import numpy as np
import pytest as pytest
from sklearn.linear_model import Perceptron as SKLPerceptron

from IMLearn.learners.classifiers import Perceptron


@dataclasses.dataclass
class PerceptronTestCase:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray = None
    include_intercept: bool = True


PERCEPTRON_TEST_CASES = [
    PerceptronTestCase(
        name="seperate between y+ and y-",
        x_train=np.array([(0, 1), (0, -1)]),
        y_train=np.array([1, -1]),
        x_test=np.array([(0, 2), (0, -2)]),
        y_test=np.array([1, -1]),
    ),
    PerceptronTestCase(
        name="seperate between x+ and x-",
        x_train=np.array([(1, 0), (-1, 0)]),
        y_train=np.array([-1, 1]),
        x_test=np.array([(2, 0), (-2, 0)]),
        y_test=np.array([-1, 1]),
    ),
    PerceptronTestCase(
        name="seperate between z+ and z-",
        x_train=np.array([(0, 0, 1), (0, 0, -1)]),
        y_train=np.array([1, -1]),
        x_test=np.array([(0, 0, 2), (0, 0, -2)]),
        y_test=np.array([1, -1]),
    ),
    # PerceptronTestCase(
    #     name="seperate between y=x+1 and y=x-1",
    #     x_train=np.array([(x, x + 1) for x in range(-10, 10, 2)] + [(x, x - 1) for x in range(-10, 10, 2)]),
    #     y_train=np.array([1] * 10 + [-1] * 10),
    #     x_test=np.array([(10, 11), (10, 9)]),
    #     y_test=np.array([1, -1]),
    # ),
]


@pytest.mark.parametrize("test_case", PERCEPTRON_TEST_CASES, ids=lambda x: x.name)
def test_perception(test_case: PerceptronTestCase):
    # arrange
    perceptron = Perceptron(include_intercept=test_case.include_intercept)

    # act
    trained_perceptron = perceptron.fit(X=test_case.x_train, y=test_case.y_train)

    # assert
    _assert_perception(result=trained_perceptron, test_case=test_case)


def _assert_perception(result: Perceptron, test_case: PerceptronTestCase):
    skl_model = SKLPerceptron(fit_intercept=test_case.include_intercept).fit(X=test_case.x_train, y=test_case.y_train)

    # assert weights
    assert np.all(result.coefs_ == np.c_[skl_model.intercept_, skl_model.coef_])

    # assert predication
    assert np.all(result.predict(test_case.x_test) == skl_model.predict(test_case.x_test))

    if test_case.y_test is not None:  # used only when it's really clear what is the expected output
        assert np.all(result.predict(test_case.x_test) == test_case.y_test)
