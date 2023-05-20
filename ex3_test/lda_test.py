import dataclasses
from typing import Tuple

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LdaSkl
from sklearn.utils import check_random_state

from IMLearn.learners.classifiers.linear_discriminant_analysis import LDA
from IMLearn.metrics import accuracy

# Data is just 6 separable points in the plane
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]], dtype="f")
y = np.array([1, 1, 1, 2, 2, 2])

# Degenerate data with only one feature (still should be separable)
X1 = np.array([[-2], [-1], [-1], [1], [1], [2]], dtype="f")

# Data that has zero variance in one dimension and needs regularization
X2 = np.array([[-3, 0], [-2, 0], [-1, 0], [-1, 0], [0, 0], [1, 0], [1, 0], [2, 0], [3, 0]])

y3 = np.array([1, 1, 2, 2, 3, 3])


@dataclasses.dataclass
class LdaPredictTestCase:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_true: np.ndarray


LDA_PREDICT_TEST_CASES = [
    LdaPredictTestCase(
        name="train data is just 6 separable points in the plane - sanity check",
        x_train=X,
        y_train=y,
        x_test=X,
        y_true=y,
    ),
    LdaPredictTestCase(
        name="check that it works with 1D data",
        x_train=X1,
        y_train=y,
        x_test=X1,
        y_true=y,
    ),
    LdaPredictTestCase(
        name="check that it works with 1D data",
        x_train=X1,
        y_train=y,
        x_test=X1,
        y_true=y,
    ),
]


def generate_train_data(samples_per_class: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(0)
    X1 = np.random.randn(samples_per_class, 2) + [0, 0]
    X2 = np.random.randn(samples_per_class, 2) + [5, 5]
    X3 = np.random.randn(samples_per_class, 2) + [5, 0]
    X = np.concatenate([X1, X2, X3], axis=0)
    y = np.array([0] * samples_per_class + [1] * samples_per_class + [2] * samples_per_class)
    return X, y


def generate_test_data(num_of_class: int, num_samples: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    if num_of_class == 0:
        X = np.random.randn(num_samples, 2) + [0, 0]
        y = np.array([0] * num_samples)
    elif num_of_class == 1:
        X = np.random.randn(num_samples, 2) + [5, 5]
        y = np.array([1] * num_samples)
    else:  # num_of_class == 2
        X = np.random.randn(num_samples, 2) + [5, 0]
        y = np.array([2] * num_samples)
    return X, y


def generate_test_and_train_data(num_of_class: int, num_samples: int = 20):
    x_train, y_train = generate_train_data(num_samples)
    x_test, y_test = generate_test_data(num_of_class, num_samples)
    return x_train, y_train, x_test, y_test


def assert_lda_fit(our_lda: LDA, X, y):
    lda_skl = LdaSkl(store_covariance=True).fit(X, y)
    assert np.all(np.allclose(our_lda.mu_, lda_skl.means_))
    assert np.all(np.allclose(our_lda.classes_, lda_skl.classes_))
    assert np.all(np.allclose(our_lda.pi_, lda_skl.priors_))
    lda_biased_cov = lda_skl.covariance_
    lda_unbiased_cov = lda_biased_cov * (len(y) / (len(y) - len(lda_skl.classes_)))
    assert np.all(np.allclose(our_lda.cov_, lda_unbiased_cov))


@pytest.fixture
def get_fitted_lda() -> callable:
    def _fit_lda(X, y):
        return LDA().fit(X, y)

    return _fit_lda


def test_lda_fit():
    # Arrange
    X, y = generate_train_data()
    # Act
    our_lda = LDA().fit(X, y)
    # Assert
    assert_lda_fit(our_lda, X, y)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_lda_fit_advenced(n_classes, get_fitted_lda):
    def generate_dataset(n_samples, centers, covariances, random_state=None):
        """Generate a multivariate normal data given some centers and covariances"""
        rng = check_random_state(random_state)
        X = np.vstack(
            [
                rng.multivariate_normal(mean, cov, size=n_samples // len(centers))
                for mean, cov in zip(centers, covariances)
            ]
        )
        y = np.hstack(
            [[clazz] * (n_samples // len(centers)) for clazz in range(len(centers))]
        )
        return X, y

    blob_centers = np.array([[0, 0], [-10, 40], [-30, 30]])[:n_classes]
    blob_stds = np.array([[[10, 10], [10, 100]]] * len(blob_centers))
    X, y = generate_dataset(
        n_samples=90000, centers=blob_centers, covariances=blob_stds, random_state=42
    )
    lda = get_fitted_lda(X, y)
    # check that the empirical means and covariances are close enough to the
    # one used to generate the data
    assert_allclose(lda.mu_, blob_centers, atol=1e-1)
    assert_allclose(lda.cov_, blob_stds[0], atol=1)
    assert np.all(lda.classes_ == np.array(range(n_classes)))

    # check that the probability of LDA are close to the theoretical probabilties
    sample = np.array([[-22, 22]])
    prob, prob_ref = calculate_expected_probability(blob_centers, blob_stds, n_classes, sample)
    lda_likelihood = lda.likelihood(sample)
    assert lda_likelihood.shape == (1, n_classes)


def test_lda_should_not_fit(get_fitted_lda):
    test_case = LdaPredictTestCase(
        name="LDA shouldn't be able to separate those",
        x_train=X,
        y_train=y3,
        x_test=X,
        y_true=y3,
    )
    lda = get_fitted_lda(test_case.x_train, test_case.y_train)
    y_pred = lda.predict(test_case.x_test)
    assert np.any(y_pred != test_case.y_true)


@pytest.mark.parametrize("num_of_class", [i for i in range(3)], ids=lambda i: f"all test data comes from class {i}")
def test_lda_predict(num_of_class):
    # Arrange
    x_train, y_train, x_test, y_test = generate_test_and_train_data(num_of_class)
    # Act
    lda = LDA().fit(x_train, y_train)
    y_pred = lda.predict(x_test)
    assert accuracy(y_pred=y_pred, y_true=y_test) == 1.0
    assert np.all(y_pred == y_test)
    assert lda.loss(x_test, y_test) == 0


@pytest.mark.parametrize("test_case", LDA_PREDICT_TEST_CASES, ids=lambda x: x.name)
def test_lda_predict_advanced(test_case: LdaPredictTestCase, get_fitted_lda):
    """
    Test LDA classification.
    This checks that LDA implements fit and predict and returns correct
    Values for simple toy data.
    """
    # arrange
    lda = get_fitted_lda(test_case.x_train, test_case.y_train)
    # act
    y_pred = lda.predict(test_case.x_test)
    # assert
    assert np.all(y_pred == test_case.y_true)


def test_lda_likelihood():
    # X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, n_clusters_per_class=1)
    X, y = generate_train_data(samples_per_class=1000)
    lda = LDA()
    lda.fit(X, y)
    likelihoods = lda.likelihood(X)
    assert likelihoods.shape == (3000, 3)
    assert np.all(likelihoods >= 0)


def calculate_expected_probability(blob_centers, blob_stds, n_classes, sample):
    # implement the method to compute the probability given in The Elements
    inv_cov = np.linalg.inv(blob_stds[0])
    alpha_k = []
    alpha_k_0 = []
    for clazz in range(len(blob_centers) - 1):
        alpha_k.append(
            np.dot(inv_cov, (blob_centers[clazz] - blob_centers[-1])[:, np.newaxis])
        )
        alpha_k_0.append(
            np.dot(
                -0.5 * (blob_centers[clazz] + blob_centers[-1])[np.newaxis, :],
                alpha_k[-1],
            )
        )

    def discriminant_func(sample, coef, intercept, clazz):
        return np.exp(intercept[clazz] + np.dot(sample, coef[clazz]))

    prob = np.array(
        [
            float(
                discriminant_func(sample, alpha_k, alpha_k_0, clazz)
                / (
                        1
                        + sum(
                    [
                        discriminant_func(sample, alpha_k, alpha_k_0, clazz)
                        for clazz in range(n_classes - 1)
                    ]
                )
                )
            )
            for clazz in range(n_classes - 1)
        ]
    )
    prob_ref = 1 - np.sum(prob)
    # check the consistency of the computed probability
    # all probabilities should sum to one
    prob_ref_2 = float(
        1 / (1 + sum([discriminant_func(sample, alpha_k, alpha_k_0, clazz) for clazz in range(n_classes - 1)]))
    )
    assert prob_ref == pytest.approx(prob_ref_2)
    return prob, prob_ref
