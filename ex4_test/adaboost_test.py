import dataclasses
from typing import Callable
from typing import List

import numpy as np
import pytest

from IMLearn import BaseEstimator
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners import AdaBoost
from IMLearn.metrics import misclassification_error
import test_utils
from exercises.adaboost_scenario import generate_data

#         ^
#    -1   |   1    1   1
# ----●-------●----●---●-->
#         |


X1 = np.array([[-1, 0], [1, 0], [2, 0], [3, 0]])
y1 = np.array([-1, 1, 1, 1])

#          ^
#          |
#          | ●   1
#          | ● ●
# ---------0-------->
#      ● ● |
#   -1   ● |
#          |
#          |

X2 = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
y2 = np.array([-1, -1, -1, 1, 1, 1])

# y3 is not linear separable, but can be solved using adaboost with decision stump
y3 = np.array([-1, -1, -1, -1, 1, 1])

# circular data from adaboost scenario
circular_train_X, circular_train_y = generate_data(n=50, noise_ratio=0)
circular_test_X, circular_test_y = generate_data(n=20, noise_ratio=0)


@dataclasses.dataclass
class AdaboostFitTestCase(test_utils.FitTestCase):
    wl: Callable[[], BaseEstimator]
    iterations: int
    expected_num_of_models: int
    expected_num_of_weights: int
    expected_signs_of_weights: List[int]


ADABOOST_FIT_TEST_CASES = [

    AdaboostFitTestCase(
        name="Simple decision tree can solve the problem should return one tree - 1",
        X=X1,
        y=y1,
        wl=lambda: DecisionStump(),
        iterations=5,
        expected_num_of_models=1,
        expected_num_of_weights=1,
        expected_signs_of_weights=[1]

        # If you are not passing this test, it might be you are getting different valid result. you can replace
        # expected_num_of_models, expected_num_of_weights, expected_signs_of_weights with the following

        # expected_num_of_models=5,
        # expected_num_of_weights=5,
        # expected_signs_of_weights=[1, 0, 0, 0, 0]
    ),
    AdaboostFitTestCase(
        name="Simple decision tree can solve the problem should return one tree - 2",
        X=X2,
        y=y2,
        wl=lambda: DecisionStump(),
        iterations=5,
        expected_num_of_models=1,
        expected_num_of_weights=1,
        expected_signs_of_weights=[1]

        # If you are not passing this test, it might be you are getting different valid result. you can replace
        # expected_num_of_models, expected_num_of_weights, expected_signs_of_weights with the following

        # expected_num_of_models=5,
        # expected_num_of_weights=5,
        # expected_signs_of_weights=[1, 0, 0, 0, 0]
    ),
    AdaboostFitTestCase(
        name="Data is not separable - should return num of iterations weight and models",
        X=X2,
        y=y3,
        wl=lambda: DecisionStump(),
        iterations=5,
        expected_num_of_models=5,
        expected_num_of_weights=5,
        expected_signs_of_weights=[1, 1, 1, 1, 1]
    ),
    AdaboostFitTestCase(
        name="Weak learner always predicts 1, should be always wrong on one sample",
        X=X1,
        y=y1,
        wl=lambda: test_utils.AlwaysPredictsOne(),
        iterations=5,
        expected_num_of_models=5,
        expected_num_of_weights=5,
        expected_signs_of_weights=[1, 0, 0, 0, 0]
    ),

    # In the forum they said that we are should not get negative scenarios with negative weights,
    # but I still think that it is a possible scenario that doesn't contradict anything.
    # If you are not passing this test, you can remove it
    AdaboostFitTestCase(
        name="Weak learner always predicts -1, should be wrong always on three samples, but weight should be negative so than the learner only wrong once",
        wl=lambda: test_utils.AlwaysPredictsMinusOne(),
        X=X1,
        y=y1,
        iterations=5,
        expected_num_of_models=5,
        expected_num_of_weights=5,
        expected_signs_of_weights=[-1, 0, 0, 0, 0]
    ),
]


@dataclasses.dataclass
class AdaboostPredictTestCase(test_utils.PredictTestCase):
    wl: Callable[[], BaseEstimator]
    iterations: int


ADABOOST_PREDICT_TEST_CASES = [
    AdaboostPredictTestCase(
        name="Weak learner always predict 1 loss should be 1/4",
        wl=lambda: test_utils.AlwaysPredictsOne(),
        iterations=5,
        X_train=X1,
        y_train=y1,
        X_test=X1,
        y_test=y1,
        expected_loss=0.25,
    ),
    AdaboostPredictTestCase(
        name="Weak learner always predict -1 loss should be 1/4",
        wl=lambda: test_utils.AlwaysPredictsMinusOne(),
        iterations=5,
        X_train=X1,
        y_train=y1,
        X_test=X1,
        y_test=y1,
        expected_loss=0.25,
    ),
]


@dataclasses.dataclass
class AdaboostPartialPredictTestCase(AdaboostPredictTestCase):
    partial_predict_iterations: int
    should_be_improved: bool


ADABOOST_PARTIAL_PREDICT_TEST_CASES = [
    AdaboostPartialPredictTestCase(
        name="Weak learner is decision stump. Predict should improve when adding learners",
        wl=lambda: DecisionStump(),
        iterations=3,
        X_train=X2,
        y_train=y3,
        X_test=X2,
        y_test=y3,
        expected_loss=0,
        partial_predict_iterations=1,
        should_be_improved=True,
    ),
    AdaboostPartialPredictTestCase(
        name="Weak learner always predict 1 loss should be 1/4. Predict should not improve when adding learners",
        wl=lambda: test_utils.AlwaysPredictsOne(),
        iterations=5,
        X_train=X1,
        y_train=y1,
        X_test=X1,
        y_test=y1,
        expected_loss=0.25,
        partial_predict_iterations=1,
        should_be_improved=False,
    ),
    AdaboostPartialPredictTestCase(
        name="Weak learner always predict -1 loss should be 1/4. Predict should not improve when adding learners",
        wl=lambda: test_utils.AlwaysPredictsMinusOne(),
        iterations=5,
        X_train=X1,
        y_train=y1,
        X_test=X1,
        y_test=y1,
        expected_loss=0.25,
        partial_predict_iterations=1,
        should_be_improved=False,
    ),
    AdaboostPartialPredictTestCase(
        name="Weak learner is decision stump and data is circular. Predict should improve when adding learners",
        wl=lambda: DecisionStump(),
        iterations=30,
        X_train=X2,
        y_train=y3,
        X_test=X2,
        y_test=y3,
        expected_loss=0,
        partial_predict_iterations=2,
        should_be_improved=True,
    ),
]


@pytest.mark.parametrize("test_case", ADABOOST_FIT_TEST_CASES, ids=lambda x: x.name)
def test_adaboost_fit(test_case: AdaboostFitTestCase):
    boost = AdaBoost(wl=test_case.wl, iterations=test_case.iterations)
    boost.fit(test_case.X, test_case.y)
    assert len(boost.models_) == test_case.expected_num_of_models
    assert len(boost.weights_) == test_case.expected_num_of_weights
    assert np.allclose(np.sum(boost.D_), 1, rtol=1.e-9)
    test_utils.compare_weights(weights=boost.weights_, expected_signs_of_weights=test_case.expected_signs_of_weights)


@pytest.mark.parametrize("test_case", ADABOOST_PREDICT_TEST_CASES, ids=lambda x: x.name)
def test_adaboost_predict(test_case: AdaboostPredictTestCase):
    boost = AdaBoost(wl=test_case.wl, iterations=test_case.iterations).fit(test_case.X_train, test_case.y_train)
    res = boost.predict(test_case.X_test)
    assert misclassification_error(y_pred=res, y_true=test_case.y_test) == test_case.expected_loss


@pytest.mark.parametrize("test_case", ADABOOST_PARTIAL_PREDICT_TEST_CASES, ids=lambda x: x.name)
def test_adaboost_partial_predict(test_case: AdaboostPartialPredictTestCase):
    boost = AdaBoost(wl=test_case.wl, iterations=test_case.iterations).fit(test_case.X_train, test_case.y_train)
    partial_predict = boost.partial_predict(test_case.X_test, T=test_case.partial_predict_iterations)
    full_predict = boost.predict(test_case.X_test)

    # calculate and validate loss
    partial_predict_error = misclassification_error(y_pred=partial_predict, y_true=test_case.y_test)
    full_predict_error = misclassification_error(y_pred=full_predict, y_true=test_case.y_test)
    assert full_predict_error == test_case.expected_loss

    # check if there was an improvement
    assert (full_predict_error < partial_predict_error) == test_case.should_be_improved


@pytest.mark.parametrize("test_case", ADABOOST_PARTIAL_PREDICT_TEST_CASES, ids=lambda x: x.name)
def test_adaboost_loss(test_case: AdaboostPartialPredictTestCase):
    boost = AdaBoost(wl=test_case.wl, iterations=test_case.iterations).fit(test_case.X_train, test_case.y_train)

    # calculate errors
    partial_loss = boost.partial_loss(test_case.X_test, test_case.y_test, T=test_case.partial_predict_iterations)
    full_loss = boost.loss(test_case.X_test, test_case.y_test)
    assert full_loss == test_case.expected_loss

    # check if there was an improvement
    assert (full_loss < partial_loss) == test_case.should_be_improved

