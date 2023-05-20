import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_array_equal

from IMLearn.learners.classifiers.gaussian_naive_bayes import GaussianNaiveBayes as GaussianNB

# Data is just 6 separable points in the plane
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
y = np.array([0, 0, 0, 1, 1, 1])


@pytest.fixture
def global_random_seed() -> int:
    return 0


def get_random_normal_x_binary_y(global_random_seed):
    # A bit more random tests
    rng = np.random.RandomState(global_random_seed)
    X1 = rng.normal(size=(10, 3))
    y1 = (rng.normal(size=10) > 0).astype(int)
    return X1, y1


def test_gnb():
    # Gaussian Naive Bayes classification.
    # This checks that GaussianNB implements fit and predict and returns
    # correct values for a simple toy dataset.

    clf = GaussianNB()
    y_pred = clf.fit(X, y).predict(X)
    assert_array_equal(y_pred, y)
    assert clf.loss(X, y) == 0


def test_gnb_prior(global_random_seed):
    # Test whether class priors are properly set.
    clf = GaussianNB().fit(X, y)
    assert_array_almost_equal(np.array([3, 3]) / 6.0, clf.pi_, 8)
    X1, y1 = get_random_normal_x_binary_y(global_random_seed)
    clf = GaussianNB().fit(X1, y1)
    # Check that the class priors sum to 1
    assert_array_almost_equal(clf.pi_.sum(), 1)


def test_gnb_naive_bayes_scale_invariance():
    # Scaling the data should not change the prediction results
    iris = load_iris()
    X, y = iris.data, iris.target
    labels = [GaussianNB().fit(f * X, y).predict(f * X) for f in [1e-10, 1, 1e10]]
    assert_array_equal(labels[0], labels[1])
    assert_array_equal(labels[1], labels[2])
