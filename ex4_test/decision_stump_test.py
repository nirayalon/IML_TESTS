import dataclasses

import numpy as np
import pytest

import test_utils
from IMLearn.learners.classifiers.decision_stump import DecisionStump


@dataclasses.dataclass
class FindThresholdDecisionStumpTestCase:
    name: str
    values: np.array
    labels: np.array
    expected_threshold_bounds: str
    expected_threshold_err: float
    sign: int = 1


FIND_THRESHOLD_DECISION_STUMP_TEST_CASES = [
    FindThresholdDecisionStumpTestCase(
        name="all above - threshold should be <= 1",
        values=np.array([1, 2, 3]),
        labels=np.array([1, 1, 1]),
        expected_threshold_bounds="<=1",
        expected_threshold_err=0,
    ),
    FindThresholdDecisionStumpTestCase(
        name="all below - threshold should be >=3",
        values=np.array([1, 2, 3]),
        labels=np.array([-1, -1, -1]),
        expected_threshold_bounds=">=3",
        expected_threshold_err=0,
    ),
    FindThresholdDecisionStumpTestCase(
        name="threshold is the second value - data is separable",
        values=np.array([1, 2, 3]),
        labels=np.array([-1, 1, 1]),
        expected_threshold_bounds="1,2",
        expected_threshold_err=0,
    ),
    FindThresholdDecisionStumpTestCase(
        name="threshold is the third value - data is not separable. there is 1 error out of 5",
        values=np.array([1, 2, 3, 4, 5]),
        labels=np.array([-1, -1, 1, -1, 1]),
        expected_threshold_bounds="2,3",
        expected_threshold_err=(1 / 5),
    ),
    FindThresholdDecisionStumpTestCase(
        name="Unordered values, all above - threshold should be <= 1",
        values=np.array([2, 1, 3]),
        labels=np.array([1, 1, 1]),
        expected_threshold_bounds="<=1",
        expected_threshold_err=0,
    ),
    FindThresholdDecisionStumpTestCase(
        name="Unordered values, all below - threshold should be >=3",
        values=np.array([3, 2, 1]),
        labels=np.array([-1, -1, -1]),
        expected_threshold_bounds=">=3",
        expected_threshold_err=0,
    ),
    FindThresholdDecisionStumpTestCase(
        name="Unordered values - data is separable",
        values=np.array([2, 1, 3]),
        labels=np.array([1, -1, 1]),
        expected_threshold_bounds="1,2",
        expected_threshold_err=0,
    ),
    FindThresholdDecisionStumpTestCase(
        name="Unordered values - data is not separable. there is 1 error out of 5",
        values=np.array([3, 2, 1, 5, 4]),
        labels=np.array([1, -1, -1, 1, -1]),
        expected_threshold_bounds="2,3",
        expected_threshold_err=(1 / 5),
    ),
]


@dataclasses.dataclass
class FitDecisionStumpTestCase(test_utils.FitTestCase):
    expected_index: int
    expected_threshold_bounds: str
    expected_sign: int


FIT_DECISION_STUMP_TEST_CASES = [
    FitDecisionStumpTestCase(
        name="best feature is the first feature",
        X=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        y=np.array([-1, -1, 1]),
        expected_index=0,
        expected_threshold_bounds="4,7",
        expected_sign=1,
    ),
    FitDecisionStumpTestCase(
        name="No Split Possible",
        X=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        y=np.array([-1, 1, -1, 1]),
        expected_index=0,
        expected_threshold_bounds="1,4",
        expected_sign=1,
    ),
    FitDecisionStumpTestCase(
        name="best feature is z axis",
        X=np.array([[0, 0, 1], [0, 0, -1], [0, 0, 3], [0, 0, -3]]),
        y=np.array([1, -1, 1, -1]),
        expected_index=2,
        expected_threshold_bounds="-1,1",
        expected_sign=1,
    ),

]


@dataclasses.dataclass
class PredictDecisionStumpTestCase(test_utils.PredictTestCase):
    pass


PREDICTION_DECISION_STUMP_TEST_CASES = [
    PredictDecisionStumpTestCase(
        name="Real threshold is 0: train and test are the same, should get 0 error",
        X_train=np.array([[-1, -2], [1, 2], ]),
        y_train=np.array([-1, 1]),
        X_test=np.array([[-1, -2], [1, 2], ]),
        y_test=np.array([-1, 1]),
        expected_loss=0,
    ),
    PredictDecisionStumpTestCase(
        name="Real threshold is 0: one error in the test - 1 sample, loss should be 1",
        X_train=np.array([[-1, -2], [1, 2], ]),
        y_train=np.array([-1, 1]),
        X_test=np.array([[-3, -4]]),
        y_test=np.array([1]),
        expected_loss=1,
    ),
    PredictDecisionStumpTestCase(
        name="Real threshold is 0: 1 error in the test - 4 sample, loss should be 0.25",
        X_train=np.array([[-1, -2], [1, 2], ]),
        y_train=np.array([-1, 1]),
        X_test=np.array([[-3, -4], [3, 4], [-5, -6], [5, 6]]),
        y_test=np.array([1, 1, -1, 1]),
        expected_loss=0.25,
    ),
    PredictDecisionStumpTestCase(
        name="threshold is 7: train and test are the same, should get 0 error",
        X_train=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        y_train=np.array([-1, -1, 1]),
        X_test=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        y_test=np.array([-1, -1, 1]),
        expected_loss=0,
    ),
    PredictDecisionStumpTestCase(
        name="threshold is 7: one error in the test - 1 sample, loss should be 1",
        X_train=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        y_train=np.array([-1, -1, 1]),
        X_test=np.array([[8, 9, 10]]),
        y_test=np.array([-1]),
        expected_loss=1,
    ),
    PredictDecisionStumpTestCase(
        name="threshold is 7: one error in the test - 2 samples, loss should be 0.5",
        X_train=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        y_train=np.array([-1, -1, 1]),
        X_test=np.array([[8, 9, 10], [9, 10, 11]]),
        y_test=np.array([-1, 1]),
        expected_loss=0.5,
    ),

]


@pytest.fixture
def decision_stump() -> DecisionStump:
    return DecisionStump()


def assert_threshold_bounds(threshold: float, expected_bounds: str) -> None:
    if expected_bounds.startswith("<="):
        d = float(expected_bounds[2:])
        assert threshold <= d
    elif expected_bounds.startswith(">="):
        d = float(expected_bounds[2:])
        assert threshold >= d
    else:
        d1, d2 = map(float, expected_bounds.split(","))
        assert d1 < threshold <= d2


@pytest.mark.parametrize("test_case", FIND_THRESHOLD_DECISION_STUMP_TEST_CASES, ids=lambda x: x.name)
def test_find_threshold_decision_tree(decision_stump: DecisionStump, test_case: FindThresholdDecisionStumpTestCase):
    threshold, threshold_err = decision_stump._find_threshold(test_case.values, test_case.labels, test_case.sign)
    assert_threshold_bounds(threshold, test_case.expected_threshold_bounds)
    assert threshold_err == test_case.expected_threshold_err


@pytest.mark.parametrize("test_case", FIT_DECISION_STUMP_TEST_CASES, ids=lambda x: x.name)
def test_fit_decision_tree(decision_stump: DecisionStump, test_case: FitDecisionStumpTestCase):
    decision_stump.fit(X=test_case.X, y=test_case.y)
    assert decision_stump.j_ == test_case.expected_index
    assert_threshold_bounds(threshold=decision_stump.threshold_, expected_bounds=test_case.expected_threshold_bounds)
    assert decision_stump.sign_ == test_case.expected_sign


@pytest.mark.parametrize("test_case", PREDICTION_DECISION_STUMP_TEST_CASES, ids=lambda x: x.name)
def test_predict_decision_tree(decision_stump: DecisionStump, test_case: PredictDecisionStumpTestCase):
    decision_stump.fit(X=test_case.X_train, y=test_case.y_train)
    assert decision_stump.loss(test_case.X_test, test_case.y_test) == test_case.expected_loss
