import numpy as np

from .budgetmanager import (
    FixedUncertaintyBudgetManager,
    VariableUncertaintyBudgetManager,
    SplitBudgetManager,
    RandomVariableUncertaintyBudgetManager, BalancedIncrementalQuantileFilter,
)
from .budgetmanager._estimated_budget_xu import EstimatedBudgetXu
from ..base import (
    BudgetManager,
    SingleAnnotatorStreamQueryStrategy,
    SkactivemlClassifier,
)
from ..utils import (
    check_type,
    call_func,
    check_budget_manager,
)


class PairedEnsembleStrategy(SingleAnnotatorStreamQueryStrategy):
    def __init__(
            self,
            budget=None,
            budget_manager=None,
            random_state=None,
            v=0.3
    ):
        super().__init__(budget=budget, random_state=random_state)
        self.budget_manager = EstimatedBudgetXu(budget=budget)
        self.v = v
        self.rng = np.random.default_rng(random_state)
        self.random_sampled = True

    def query(
            self,
            candidates,
            clf,
            X=None,
            y=None,
            sample_weight=None,
            fit_clf=False,
            return_utilities=False,
    ):

        # Generate random variable to decide whether random or uncertainty sampling
        u = self.rng.uniform(0, 1)
        utility = None
        if u < self.v:
            r = self.rng.uniform(0, 1)
            queried_indices = self.budget_manager.query_by_random(r)
            if len(queried_indices) > 0:
                self.random_sampled = True
        else:
            # Calcluate margin uncertainty
            proba_L = sorted(clf.predict_proba(candidates)[0])
            utility = proba_L[-1] - proba_L[-2]
            # Check if labeling
            queried_indices = self.budget_manager.query_by_utility(utility)
            if len(queried_indices) > 0:
                self.random_sampled = False

        return queried_indices, utility

    def update(
            self, candidates, queried_indices, budget_manager_param_dict=None
    ):
        """Updates the budget manager.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape
        (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from candidates have been queried.

        budget_manager_param_dict : kwargs, optional (default=None)
            Optional kwargs for budgetmanager.

        Returns
        -------
        self : StreamProbabilisticAL
            PALS returns itself, after it is updated.
        """
        # check if a budgetmanager is set

        return self
