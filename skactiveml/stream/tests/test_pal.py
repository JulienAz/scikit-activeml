import unittest
import numpy as np

from sklearn.datasets import make_classification

from ...classifier import PWC
from .._pal import PAL


class TestPAL(unittest.TestCase):
    def setUp(self):
        # initialise valid data to test uncertainty parameters
        rand = np.random.RandomState(0)
        stream_length = 1000
        train_init_size = 10
        X, y = make_classification(
            n_samples=stream_length + train_init_size,
            random_state=rand.randint(2 ** 31 - 1),
            shuffle=True,
        )

        self.X = X[:train_init_size, :]
        self.X_cand = X[train_init_size:, :]
        self.y = y[:train_init_size]
        self.clf = PWC()
        self.kwargs = dict(
            X_cand=self.X_cand, clf=self.clf, X=self.X, y=self.y
        )

    def test_pal(self):
        # init param test
        self._test_init_param_budget_manager(PAL)
        self._test_init_param_prior(PAL)
        self._test_init_param_m_max(PAL)
        self._test_init_param_random_state(PAL)

        # query param test
        self._test_query_param_clf(PAL)
        self._test_query_param_X_cand(PAL)
        self._test_query_param_X(PAL)
        self._test_query_param_y(PAL)
        self._test_query_param_sample_weight(PAL)
        self._test_query_param_return_utilities(PAL)

    def _test_init_param_budget_manager(self, query_strategy_name):
        # budget_manager must be defined as an object of an budget manager
        # class
        query_strategy = query_strategy_name(budget_manager=[])
        self.assertRaises(TypeError, query_strategy.query, **(self.kwargs))

    def _test_init_param_prior(self, query_strategy_name):
        # prior must be defined as a float with a range of: 0 < prior
        query_strategy = query_strategy_name(prior="string")
        self.assertRaises(TypeError, query_strategy.query, **(self.kwargs))
        query_strategy = query_strategy_name(prior=0.0)
        self.assertRaises(ValueError, query_strategy.query, **(self.kwargs))
        query_strategy = query_strategy_name(prior=-1.0)
        self.assertRaises(ValueError, query_strategy.query, **(self.kwargs))

    def _test_init_param_m_max(self, query_strategy_name):
        # m_max must be defined as a integer greater than 0
        query_strategy = query_strategy_name(m_max="string")
        self.assertRaises(TypeError, query_strategy.query, **(self.kwargs))
        query_strategy = query_strategy_name(m_max=0.1)
        self.assertRaises(TypeError, query_strategy.query, **(self.kwargs))
        query_strategy = query_strategy_name(m_max=0)
        self.assertRaises(ValueError, query_strategy.query, **(self.kwargs))
        query_strategy = query_strategy_name(m_max=-1)
        self.assertRaises(ValueError, query_strategy.query, **(self.kwargs))

    def _test_init_param_random_state(self, query_strategy_name):
        query_strategy = query_strategy_name(random_state="string",)
        self.assertRaises(ValueError, query_strategy.query, **(self.kwargs))

    def _test_query_param_X_cand(self, query_strategy_name):
        # X_cand must be defined as a two dimensinal array
        query_strategy = query_strategy_name()
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=1,
            clf=self.clf,
            X=self.X,
            y=self.y,
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=None,
            clf=self.clf,
            X=self.X,
            y=self.y,
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=np.ones(5),
            clf=self.clf,
            X=self.X,
            y=self.y,
        )

    def _test_query_param_clf(self, query_strategy_name):
        # clf must be defined as a classifier
        query_strategy = query_strategy_name()
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf="string",
            X=self.X,
            y=self.y,
        )
        query_strategy = query_strategy_name()
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=1,
            X=self.X,
            y=self.y,
        )

    def _test_query_param_X(self, query_strategy_name):
        # X must be defined as a two dimensinal array and must be equal in
        # length to y
        query_strategy = query_strategy_name()
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=1,
            y=self.y,
        )
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=None,
            y=self.y,
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=np.ones(5),
            y=self.y,
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X[1:],
            y=self.y,
        )

    def _test_query_param_y(self, query_strategy_name):
        # y must be defined as a one Dimensional array and must be equal in
        # length to X
        query_strategy = query_strategy_name()
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=1,
        )
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=None,
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=self.y[1:],
        )

    def _test_query_param_sample_weight(self, query_strategy_name):
        # sample weight needs to be a list that can be convertet to float
        # equal in size of y
        query_strategy = query_strategy_name()
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=self.y[1:],
            sample_weight="string",
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=self.y[1:],
            sample_weight=["string", "numbers", "test"],
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=self.y[1:],
            sample_weight=[1],
        )

    def _test_query_param_return_utilities(self, query_strategy_name):
        # return_utilities needs to be a boolean
        query_strategy = query_strategy_name()
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=self.y[1:],
            return_utilities="string",
        )
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=self.y[1:],
            return_utilities=1,
        )
