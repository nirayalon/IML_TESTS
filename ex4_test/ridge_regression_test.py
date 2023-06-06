import dataclasses

import numpy as np
import pytest as pytest
from sklearn.linear_model import Ridge

import test_utils
from IMLearn.learners.regressors import RidgeRegression

y1 = np.array([1, 2, 3])

X1 = np.array([[1, 2], [3, 4], [5, 6]])
X2 = np.array([[1, 2], [1, 4], [1, 6]])


@dataclasses.dataclass
class RidgeFitTestCase(test_utils.FitTestCase):
    lam: float
    include_intercept: bool
    expected_weights: np.array


RIDGE_FIT_TEST_CASES = [
    RidgeFitTestCase(
        name="Lambda=0, solution should be like OLS",
        lam=0,
        include_intercept=True,
        X=X2,
        y=y1,
        expected_weights=np.array([0, 0, 0.5])
    ),
    RidgeFitTestCase(
        name="Lambda is very big, weights should be close to zero, with intercept",
        lam=1e10,
        include_intercept=True,
        X=X1,
        y=y1,
        expected_weights=np.array([2, 0, 0])
    ),
    RidgeFitTestCase(
        name="Lambda is very big, weights should be close to zero, with no intercept",
        lam=1e10,
        include_intercept=False,
        X=X1,
        y=y1,
        expected_weights=np.array([0, 0])
    ),
    RidgeFitTestCase(
        name="Test case with lambda = 0 without intercept",
        lam=0.0,
        include_intercept=False,
        X=X1,
        y=y1,
        expected_weights=np.array([0, 0.5])
    ),
    RidgeFitTestCase(
        name="Test case with lambda = 2 with intercept",
        lam=2,
        include_intercept=True,
        X=X2,
        y=y1,
        expected_weights=np.array([0.4, 0, 0.4])
    ),

    # fit_with_single_sample
]


@dataclasses.dataclass
class RidgePredictTestCase(test_utils.PredictTestCase):
    lam: float
    include_intercept: bool


RIDGE_PREDICT_TEST_CASES = [
    RidgePredictTestCase(
        name="lambda=0, Test data = train data, expected loss = 0",
        X_train=X2,
        y_train=y1,
        X_test=X2,
        y_test=y1,
        lam=0,
        include_intercept=True,
        expected_loss=0
    ),
    RidgePredictTestCase(
        name="lambda=0, Test data = train data, expected loss = 0",
        X_train=X2,
        y_train=y1,
        X_test=X2,
        y_test=y1,
        lam=0,
        include_intercept=False,
        expected_loss=0
    ),
    RidgePredictTestCase(
        name="lambda=1e10, Test data = train data, expected loss = 2/3",
        X_train=X2,
        y_train=y1,
        X_test=X2,
        y_test=y1,
        lam=1e10,
        include_intercept=True,
        expected_loss=2 / 3
    ),
    RidgePredictTestCase(
        name="lambda=1e10, Test data = train data, expected loss = 14/3",
        X_train=X2,
        y_train=y1,
        X_test=X2,
        y_test=y1,
        lam=1e10,
        include_intercept=False,
        expected_loss=14 / 3
    ),
    RidgePredictTestCase(
        name="lambda=2, Test data = train data, expected loss = 0",
        X_train=X2,
        y_train=y1,
        X_test=X2,
        y_test=y1,
        lam=2,
        include_intercept=True,
        expected_loss=0.08 / 3
    ),
]


@pytest.mark.parametrize("test_case", RIDGE_FIT_TEST_CASES, ids=lambda x: x.name)
def test_ridge_fit(test_case: RIDGE_FIT_TEST_CASES):
    # arrange
    ridge = RidgeRegression(lam=test_case.lam, include_intercept=test_case.include_intercept)

    # act
    ridge.fit(test_case.X, test_case.y)

    # assert
    assert np.allclose(ridge.coefs_, test_case.expected_weights, rtol=1.e-3)

    if test_case.include_intercept:
        assert len(ridge.coefs_) == test_case.X.shape[1] + 1
    else:
        assert len(ridge.coefs_) == test_case.X.shape[1]


@pytest.mark.parametrize("include_intercept", [True, False],
                         ids=lambda include_intercept: f"include intercept:{include_intercept}")
@pytest.mark.parametrize("lam", [i for i in np.linspace(0, 2, 5)], ids=lambda lam: f"compare skl fit with lambda={lam}")
def test_ridge_fit_compare_to_skl_implementation(lam, include_intercept):
    ridge = RidgeRegression(lam=lam, include_intercept=include_intercept).fit(X2, y1)
    ridge_skl = Ridge(alpha=lam, fit_intercept=include_intercept, solver="svd").fit(X2, y1)

    skl_ridge_coefs = [ridge_skl.intercept_] + ridge_skl.coef_.tolist() if include_intercept else ridge_skl.coef_

    assert np.allclose(ridge.coefs_, skl_ridge_coefs, rtol=1.e-3)


@pytest.mark.parametrize("test_case", RIDGE_PREDICT_TEST_CASES, ids=lambda x: x.name)
def test_ridge_predict(test_case: RidgePredictTestCase):
    ridge = RidgeRegression(lam=test_case.lam, include_intercept=test_case.include_intercept).fit(test_case.X_train,
                                                                                                  test_case.y_train)
    ridge_loss = ridge.loss(test_case.X_test, test_case.y_test)
    assert np.isclose(test_case.expected_loss, ridge_loss)
