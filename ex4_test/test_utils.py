import dataclasses
from typing import List
from typing import NoReturn

import numpy as np

from IMLearn import BaseEstimator
from IMLearn.metrics import misclassification_error


@dataclasses.dataclass
class FitTestCase:
    name: str
    X: np.array
    y: np.array


@dataclasses.dataclass
class PredictTestCase:
    name: str
    X_train: np.array
    y_train: np.array
    X_test: np.array
    y_test: np.array
    expected_loss: float


def compare_weights(expected_signs_of_weights: List[int], weights: np.array):
    assert len(expected_signs_of_weights) == len(weights)

    for i in range(len(weights)):
        if expected_signs_of_weights[i] == 0:
            assert np.allclose(weights[i], 0, rtol=1.e-15), "Expected weight {} to be zero".format(weights[i])
        elif expected_signs_of_weights[i] < 0:
            assert np.sign(weights[i]) == -1, "Expected weight {} to be negative".format(weights[i])
        elif expected_signs_of_weights[i] > 0:
            assert np.sign(weights[i]) == 1, "Expected weight {} to be positive".format(weights[i])


class AlwaysPredictsOne(BaseEstimator):
    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        pass

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[0])

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return misclassification_error(y_true=y, y_pred=self._predict(X))


class AlwaysPredictsMinusOne(BaseEstimator):
    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        pass

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return -np.ones(X.shape[0])

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return misclassification_error(y_true=y, y_pred=self._predict(X))
