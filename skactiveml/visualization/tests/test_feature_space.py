import unittest
import numpy as np
import os

from matplotlib import pyplot as plt
from matplotlib import testing
from matplotlib.testing.compare import compare_images
from sklearn.datasets import make_classification

from skactiveml.classifier import PWC
from skactiveml import visualization
from skactiveml.pool import UncertaintySampling
from skactiveml.visualization._feature_space import plot_decision_boundary, \
    plot_utility


class TestFeatureSpace(unittest.TestCase):

    def setUp(self):
        self.path_prefix = os.path.dirname(visualization.__file__) + \
                           '/tests/images/'
        np.random.seed(0)
        self.X, self.y = make_classification(n_features=2, n_redundant=0,
                                             random_state=0)
        train_indices = np.random.randint(0, len(self.X), size=20)
        cand_indices = np.setdiff1d(np.arange(len(self.X)), train_indices)
        self.X_train = self.X[train_indices]
        self.y_train = self.y[train_indices]
        self.X_cand = self.X[cand_indices]
        self.clf = PWC()
        self.clf.fit(self.X_train, self.y_train)
        self.qs = UncertaintySampling(clf=self.clf)
        self.qs_dict = {'X': self.X_train, 'y': self.y_train}

        x1_min = min(self.X[:, 0])
        x1_max = max(self.X[:, 0])
        x2_min = min(self.X[:, 1])
        x2_max = max(self.X[:, 1])
        self.bound = [[x1_min, x2_min], [x1_max, x2_max]]

        self.cmap = 'jet'

        testing.set_font_settings_for_testing()
        testing.set_reproducibility_for_testing()
        testing.setup()

    # Tests for plot_decision_boundary function
    def test_decision_boundary_clf(self):
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.qs,
                          bound=self.bound)

    def test_decision_boundary_bound(self):
        self.assertRaises(ValueError, plot_decision_boundary, clf=self.clf,
                          bound=[0, 0, 1, 1])

    def test_decision_boundary_res(self):
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.clf,
                          bound=self.bound, res='string')

    def test_decision_boundary_ax(self):
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.clf,
                          bound=self.bound, ax=3)

    def test_decision_boundary_confidence(self):
        self.assertRaises(ValueError, plot_decision_boundary, clf=self.clf,
                          bound=self.bound, confidence=0.0)
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.clf,
                          bound=self.bound, confidence='string')

    def test_decision_boundary_cmap(self):
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.clf,
                          bound=self.bound, cmap=4)

    def test_decision_boundary_boundary_dict(self):
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.clf,
                          bound=self.bound, boundary_dict='string')
        plot_decision_boundary(clf=self.clf, bound=self.bound,
                               boundary_dict={'colors': 'r'})

    def test_decision_boundary_confidence_dict(self):
        self.assertRaises(TypeError, plot_decision_boundary, clf=self.clf,
                          bound=self.bound, confidence_dict='string')
        plot_decision_boundary(clf=self.clf, bound=self.bound,
                               confidence_dict={'linestyles': ':'})

    # Tests for plot_utility function
    def test_utility_qs(self):
        self.assertRaises(TypeError, plot_utility, qs=self.clf,
                          qs_dict=self.qs_dict, bound=self.bound)

    def test_utility_qs_dict(self):
        self.assertRaises(TypeError, plot_utility, qs=self.qs,
                          qs_dict={0, 1, 2}, bound=self.bound)

        qs_dict = self.qs_dict
        qs_dict['X_cand'] = []
        self.assertRaises(ValueError, plot_utility, qs=self.qs,
                          qs_dict=qs_dict, bound=self.bound)

    def test_utility_X_cand(self):
        self.assertRaises(ValueError, plot_utility, qs=self.qs,
                          qs_dict=self.qs_dict)

    def test_utility_res(self):
        self.assertRaises(ValueError, plot_utility, qs=self.qs,
                          qs_dict=self.qs_dict, bound=self.bound, res=-3)

    def test_utility_ax(self):
        self.assertRaises(TypeError, plot_utility, qs=self.qs,
                          qs_dict=self.qs_dict, bound=self.bound, ax=2)

    def test_utility_contour_dict(self):
        self.assertRaises(TypeError, plot_utility, qs=self.qs,
                          qs_dict=self.qs_dict, bound=self.bound,
                          contour_dict='string')
        plot_utility(qs=self.qs, qs_dict=self.qs_dict, bound=self.bound,
                     contour_dict={'linestyles': '.'})

    # Graphical tests
    def test_no_candidates(self):
        plot_utility(self.qs, {'X': self.X_train, 'y': self.y_train},
                     bound=self.bound)
        plt.scatter(self.X_cand[:, 0], self.X_cand[:, 1], c='k', marker='.')
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train,
                    cmap=self.cmap, alpha=.9, marker='.')
        plot_decision_boundary(self.clf, self.bound, cmap=self.cmap)

        plt.savefig(self.path_prefix + 'dec_bound_wo_cand.pdf')
        plt.cla()
        comparison = compare_images(self.path_prefix +
                                    'dec_bound_wo_cand_base.pdf',
                                    self.path_prefix + 'dec_bound_wo_cand.pdf',
                                    tol=0)
        self.assertIsNone(comparison)

    def test_with_candidates(self):
        plot_utility(self.qs, {'X': self.X_train, 'y': self.y_train},
                     X_cand=self.X_cand)
        plt.scatter(self.X[:, 0], self.X[:, 1], c='k', marker='.')
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train,
                    cmap=self.cmap, alpha=.9, marker='.')
        plot_decision_boundary(self.clf, self.bound, cmap=self.cmap)

        plt.savefig(self.path_prefix + 'dec_bound_w_cand.pdf')
        plt.cla()
        comparison = compare_images(self.path_prefix +
                                    'dec_bound_w_cand_base.pdf',
                                    self.path_prefix + 'dec_bound_w_cand.pdf',
                                    tol=0)
        self.assertIsNone(comparison)

    def test_multi_class(self):
        X, y = make_classification(n_features=2, n_redundant=0, random_state=0,
                                   n_classes=3, n_clusters_per_class=1)
        train_indices = np.random.randint(0, len(X), size=20)
        cand_indices = np.setdiff1d(np.arange(len(X)), train_indices)
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_cand = X[cand_indices]
        clf = PWC()
        clf.fit(X_train, y_train)
        qs = UncertaintySampling(clf=clf)
        bound = [[min(X[:, 0]), min(X[:, 1])], [max(X[:, 0]), max(X[:, 1])]]

        plot_utility(qs, {'X': X_train, 'y': y_train}, bound=bound)
        plt.scatter(X_cand[:, 0], X_cand[:, 1], c='k', marker='.')
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                    cmap=self.cmap, alpha=.9, marker='.')
        plot_decision_boundary(clf, bound=bound, cmap=self.cmap, res=101)
        plt.savefig(self.path_prefix + 'dec_bound_multiclass.pdf')
        plt.cla()
        comparison = compare_images(self.path_prefix +
                                    'dec_bound_multiclass_base.pdf',
                                    self.path_prefix +
                                    'dec_bound_multiclass.pdf',
                                    tol=0)
        self.assertIsNone(comparison)
