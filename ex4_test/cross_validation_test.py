import dataclasses
from typing import Callable

import numpy as np
import pytest

from IMLearn import BaseEstimator
from IMLearn.metrics import mean_square_error
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate


# Dummy BaseEstimator class for testing
class DummyEstimator(BaseEstimator):
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        return

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return misclassification_error(y, self.predict(X))


class DummyRegressionEstimator(BaseEstimator):
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        return

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return 100 * np.ones(len(X))

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return mean_square_error(y, self.predict(X))


@dataclasses.dataclass
class CrossValidationTestCase:
    name: str
    estimator: BaseEstimator
    cv: int
    X: np.array
    y: np.array
    scoring: Callable
    expected_train_score: float
    expected_validation_score: float


CROSS_VALIDATION_TEST_CASES = [
    CrossValidationTestCase(
        name="Always predicts zero, score is , cv=3",
        estimator=DummyEstimator(),
        cv=3,
        X=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        y=np.array([0, 1, 0, 1, 0]),
        scoring=misclassification_error,
        expected_train_score=np.mean([1 / 3, 1 / 3, 0.5]),
        expected_validation_score=np.mean([0.5, 0.5, 0]),
    ),
    CrossValidationTestCase(
        name="Always predicts zero, cv=2",
        estimator=DummyEstimator(),
        cv=2,
        X=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        y=np.array([0, 1, 0, 1, 0]),
        scoring=misclassification_error,
        expected_train_score=np.mean([0.5, 1 / 3]),
        expected_validation_score=np.mean([1 / 3, 0.5]),
    ),
    # CrossValidationTestCase(
    #     name="Always predicts 100, cv=4",
    #     estimator=DummyEstimator(),
    #     cv=2,
    #     X=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
    #     y=np.array([0, 1, 0, 1, 0]),
    #     scoring=misclassification_error,
    #     expected_train_score=np.mean([0.5, 1 / 3]),
    #     expected_validation_score=np.mean([1 / 3, 0.5]),
    # ),
]


@pytest.mark.parametrize("test_case", CROSS_VALIDATION_TEST_CASES, ids=lambda x: x.name)
def test_cross_validation(test_case: CrossValidationTestCase):
    train_score, validation_score = cross_validate(
        estimator=test_case.estimator,
        cv=test_case.cv,
        X=test_case.X,
        y=test_case.y,
        scoring=test_case.scoring,
    )
    assert np.isclose(train_score, test_case.expected_train_score)
    assert np.isclose(validation_score, test_case.expected_validation_score)
