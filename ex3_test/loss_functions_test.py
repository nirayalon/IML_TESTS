import dataclasses

import numpy as np
import pytest as pytest

from IMLearn.metrics import accuracy
from IMLearn.metrics import misclassification_error


@dataclasses.dataclass
class MisclassificationErrorTestCase:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    expected_result: float
    is_normalized: bool = True


MISCLASSIFICATION_ERROR_TEST_CASES = [
    MisclassificationErrorTestCase(
        name="No error at all and is normalized, should return 0",
        y_true=np.array([1, 1, 1, 1]),
        y_pred=np.array([1, 1, 1, 1]),
        expected_result=0
    ),
    MisclassificationErrorTestCase(
        name="No error at all and is not normalized, should return 0",
        y_true=np.array([1, 1, 1, 1]),
        y_pred=np.array([1, 1, 1, 1]),
        expected_result=0,
        is_normalized=False
    ),
    MisclassificationErrorTestCase(
        name="100% error and is normalized, should return 1",
        y_true=np.array([1, 1, 1, 1]),
        y_pred=np.array([-1, -1, -1, -1]),
        expected_result=1
    ),
    MisclassificationErrorTestCase(
        name="100% error and is not normalized, should return 4",
        y_true=np.array([1, 1, 1, 1]),
        y_pred=np.array([-1, -1, -1, -1]),
        expected_result=4,
        is_normalized=False
    ),
    MisclassificationErrorTestCase(
        name="50% error and is normalized, should return 1",
        y_true=np.array([-1, -1, 1, 1]),
        y_pred=np.array([-1, -1, -1, -1]),
        expected_result=0.5
    ),
    MisclassificationErrorTestCase(
        name="50% error and is not normalized, should return 2",
        y_true=np.array([-1, -1, 1, 1]),
        y_pred=np.array([-1, -1, -1, -1]),
        expected_result=2,
        is_normalized=False
    ),
]


@dataclasses.dataclass
class AccuracyTestCase:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    expected_result: float


ACCURACY_ERROR_TEST_CASES = [
    AccuracyTestCase(
        name="No error at all, should return 1",
        y_true=np.array([1, 1, 1, 1]),
        y_pred=np.array([1, 1, 1, 1]),
        expected_result=1
    ),
    AccuracyTestCase(
        name="100% error, should return 0",
        y_true=np.array([1, 1, 1, 1]),
        y_pred=np.array([-1, -1, -1, -1]),
        expected_result=0
    ),
    AccuracyTestCase(
        name="50% error, should return 0.5",
        y_true=np.array([-1, -1, 1, 1]),
        y_pred=np.array([-1, -1, -1, -1]),
        expected_result=0.5
    ),
]


@pytest.mark.parametrize("test_case", MISCLASSIFICATION_ERROR_TEST_CASES, ids=lambda x: x.name)
def test_misclassification_error(test_case: MisclassificationErrorTestCase):
    result = misclassification_error(test_case.y_true, test_case.y_pred, test_case.is_normalized)
    assert result == test_case.expected_result


@pytest.mark.parametrize("test_case", ACCURACY_ERROR_TEST_CASES, ids=lambda x: x.name)
def test_accuracy(test_case: AccuracyTestCase):
    result = accuracy(test_case.y_true, test_case.y_pred)
    assert result == test_case.expected_result
