from copy import deepcopy

import numpy as np
from sklearn.base import clone
from sklearn.metrics import pairwise_kernels
from sklearn.utils.validation import check_is_fitted

from ..base import SingleAnnotPoolBasedQueryStrategy, SkactivemlClassifier
from ..classifier import PWC
from ..utils import check_type, is_labeled, simple_batch, check_cost_matrix, \
    MISSING_LABEL, check_equal_missing_label, unlabeled_indices, is_unlabeled


class ExpectedErrorReduction(SingleAnnotPoolBasedQueryStrategy):
    """Abstract class for Expected Error Reduction (EER) algorithms

    This class implements the basic workflow of EER algorithms containing:
     - determining ever candidates x label pair and simulate its outcome
       in the classifier by simulating it
     - determining some kind of risk for the new classifier

    These structure has been used by e.g.:
     - Roy, N., & McCallum, A. (2001). Toward optimal active learning through
       monte carlo estimation of error reduction. ICML, pp. 441-448.
     - Kapoor, A., Horvitz, E., & Basu, S. (2007). Selective Supervision:
       Guiding Supervised Learning with Decision-Theoretic Active Learning.
       IJCAI, pp. 877-882.
     - Margineantu, D. D. (2005). Active cost-sensitive learning.
       IJCAI, pp. 1622-1623.
     - Joshi, A. J., Porikli, F., & Papanikolopoulos, N. P. (2012). Scalable
       active learning for multiclass image classification.
       IEEE TrPAMI, 34(11), pp. 2259-2273.

    Parameters
    ----------
    enforce_mapping : bool
        If True, an exception is raised when no exact mapping between
        instances in `X` and instances in `candidates` can be determined.
    cost_matrix: array-like, shape (n_classes, n_classes), optional
    (default=None)
        Cost matrix with `cost_matrix[i,j]` defining the cost of predicting
        class `j` for a sample with the actual class `i`.
        Used for misclassification loss and ignored for log loss.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : numeric or np.random.RandomState
        The random state to use.

    References
    ----------
    [1] Roy, N., & McCallum, A. (2001). Toward optimal active learning through
        monte carlo estimation of error reduction. ICML, (pp. 441-448).
    [2] Joshi, A. J., Porikli, F., & Papanikolopoulos, N. P. (2012).
        Scalable active learning for multiclass image classification.
        IEEE TrPAMI, 34(11), pp. 2259-2273.
    [3] Margineantu, D. D. (2005). Active cost-sensitive learning.
        In IJCAI (Vol. 5, pp. 1622-1623).
    """

    def __init__(self, enforce_mapping, cost_matrix=None,
                 missing_label=MISSING_LABEL, random_state=None):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.cost_matrix = cost_matrix
        self._enforce_mapping = enforce_mapping

    def query(self, X, y, clf, fit_clf=True, ignore_partial_fit=True,
              sample_weight=None,
              candidates=None, sample_weight_candidates=None,
              X_eval=None, sample_weight_eval=None,
              batch_size=1, return_utilities=False):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        clf : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
        fit_clf : bool, optional (default=True)
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        ignore_partial_fit : bool, optional (default=True)
            Relevant in cases where `clf` implements `partial_fit`. If True,
            the `partial_fit` function is ignored and `fit` is used instead.
        sample_weight : array-like of shape (n_samples), optional (default=None)
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
        sample_weight_candidates : array-like of shape (n_candidates),
            optional (default=None)
            Weights of candidates samples in `candidates` if candidates are
            directly given (i.e., candidates.ndim > 1). Otherwise weights for
            candidates are given in `sample_weight`.
        X_eval : array-like of shape (n_eval_samples, n_features),
            optional (default=None).
            Unlabeled evalaution data set that is used for estimating the risk.
            Not applicable for all EER methods.
        sample_weight_eval : array-like of shape (n_eval_samples),
            optional (default=None)
            Weights of evaluation samples in `X_eval` if given. Used to weight
            the importance of samples when estimating the risk.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or
            numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        """
        (
            X, y, sample_weight, clf, candidates, sample_weight_candidates,
            X_eval, sample_weight_eval, batch_size, return_utilities
        ) = self._validate_data(
            X, y, sample_weight, clf, candidates, sample_weight_candidates,
            X_eval, sample_weight_eval, batch_size, return_utilities,
            reset=True, check_X_dict=None
        )

        self._validate_init_params()

        X_cand, mapping = self._transform_candidates(
            candidates, X, y, enforce_mapping=self._enforce_mapping
        )

        X_full, y_full, w_full, w_eval, idx_train, idx_cand, idx_eval = \
            self._concatenate_samples(X, y, sample_weight,
                                      candidates, sample_weight_candidates,
                                      X_eval, sample_weight_eval)

        # Initialize classifier that works with indices to improve readability
        uclf = UpdateIndexClassifier(
            clf, X_full, y_full, w_full, fit_clf=fit_clf,
            ignore_partial_fit=ignore_partial_fit,
            missing_label=self.missing_label_
        )

        # Fit the classifier.
        uclf = self._precompute_and_fit_clf(uclf, X_full, y_full, idx_train,
                                            idx_cand, idx_eval)
        # Compute class-membership probabilities of candidate samples
        probs_cand = uclf.predict_proba(idx_cand)

        # Check cost matrix.
        classes = uclf.get_classes()
        self._validate_cost_matrix(len(classes))

        # Storage for computed errors per candidate sample
        errors = np.zeros([len(X_cand), len(classes)])

        # Iterate over candidate samples
        for i_cx, idx_cx in enumerate(idx_cand):
            # Simulate acquisition of label for each candidate sample and class
            for i_cy, cy in enumerate(classes):
                errors[i_cx, i_cy] = self._estimate_error_for_candidate(
                    uclf, [idx_cx], [cy], idx_train, idx_cand, idx_eval, w_eval
                )

        utilities_cand = np.sum(probs_cand * errors, axis=1)

        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand

        return simple_batch(utilities, self.random_state_,
                            batch_size=batch_size,
                            return_utilities=return_utilities)

    def _validate_init_params(self):
        """Function used to evaluate parameters of the `__init__` function that
        are not part of the abstract class to avoid redundancies.
        """
        pass

    def _precompute_and_fit_clf(self, uclf, X_full, y_full, idx_train, idx_cand,
                                idx_eval):
        uclf.fit(idx_train)
        return uclf

    def _estimate_error_for_candidate(self, uclf, idx_cx, cy, idx_train,
                                      idx_cand, idx_eval, w_eval):
        raise NotImplementedError('Error estimation method must be implemented'
                                  'by the query strategy.')

    def _validate_data(self, X, y, sample_weight, clf, candidates,
                       sample_weight_candidates, X_eval, sample_weight_eval,
                       batch_size, return_utilities, reset=True,
                       check_X_dict=None):

        # Validate input parameters.
        X, y, candidates, batch_size, return_utilities = \
            super()._validate_data(
                X, y, candidates, batch_size, return_utilities, reset=reset,
                check_X_dict=check_X_dict
            )

        # Validate classifier type.
        check_type(clf, 'clf', SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label_)

        # Check if candidates are samples if sample_weight_candidates is set
        if (candidates is None or candidates.ndim == 1) and \
                sample_weight_candidates is not None:
            raise ValueError('Attribute `sample_weight_candidates` can only'
                             'be None if `candidates` consists of samples.')

        # Check if X_eval is set if sample_weight_eval is set
        if X_eval is None and sample_weight_eval is not None:
            raise ValueError('If `X_eval` is None, `sample_weight_eval` must'
                             'also be None')

        # TODO: test sample weight_eval - length + column

        return X, y, sample_weight, clf, candidates, sample_weight_candidates,\
               X_eval, sample_weight_eval, batch_size, return_utilities,


    def _validate_cost_matrix(self, n_classes):

        cost_matrix = 1 - np.eye(n_classes) if self.cost_matrix is None \
            else self.cost_matrix
        self.cost_matrix_ = check_cost_matrix(cost_matrix, n_classes)

    def _concatenate_samples(self, X, y, sample_weight,
                             candidates, sample_weight_candidates,
                             X_eval, sample_weight_eval):

        # Concatenate samples
        X_full = X
        y_full = y
        w_full = sample_weight
        idx_train = np.arange(len(X))
        idx_unld = unlabeled_indices(y, self.missing_label_)

        if candidates is None:
            idx_cand = idx_unld
        elif candidates.ndim == 1:
            idx_cand = candidates
        else:
            X_full = np.concatenate([X_full, candidates], axis=0)
            y_full = np.concatenate([y_full, np.full(len(candidates), np.nan)],
                                    axis=0)
            if not (w_full is None and sample_weight_candidates is None):
                w_full = np.concatenate([w_full, sample_weight_candidates],
                                        axis=0)
            idx_cand = np.arange(len(X), len(X_full))

        if X_eval is None:
            idx_eval = idx_train
            if sample_weight_eval is None:
                w_eval = np.ones(len(X_full))
            else:
                if len(sample_weight_eval) != len(idx_eval):
                    raise ValueError('If `sample_weight_eval` is set but '
                                     '`X_eval` is None, then it should have '
                                     'same size as `X`')
                w_eval = np.zeros(len(X_full))
                w_eval[idx_eval] = sample_weight_eval
        else:
            X_full = np.concatenate([X_full, X_eval], axis=0)
            y_full = np.concatenate([y_full, np.full(len(X_eval), np.nan)],
                                    axis=0)
            idx_eval = np.arange(len(X_full) - len(X_eval), len(X_full))
            if sample_weight_eval is None:
                w_eval = np.ones(len(X_full))
            else:
                if len(sample_weight_eval) != len(idx_eval):
                    raise ValueError('If `sample_weight_eval` and `X_eval` '
                                     'are set, then `sample_weight_eval` '
                                     'should have len(X_eval)')
                w_eval = np.ones(len(X_full))
                w_eval[idx_eval] = sample_weight_eval

        return X_full, y_full, w_full, w_eval, idx_train, idx_cand, idx_eval

    def _risk_estimation(self, prob_true, prob_pred, cost_matrix,
                         sample_weight):
        if prob_true.ndim == 1 and prob_pred.ndim == 1:
            cost_est = cost_matrix[prob_true, :][len(prob_true), prob_pred]
            return np.sum(sample_weight * cost_est)
        elif prob_true.ndim == 1 and prob_pred.ndim == 2:
            cost_est = cost_matrix[prob_true, :]
            return np.sum(sample_weight[:, np.newaxis] *
                          prob_pred * cost_est[np.newaxis, :])
        elif prob_true.ndim == 2 and prob_pred.ndim == 1:
            cost_est = cost_matrix[:, prob_pred].T
            return np.sum(sample_weight[:, np.newaxis] *
                          prob_true * cost_est[np.newaxis, :])
        else:
            prob_mat = prob_true[:, :, np.newaxis]@prob_pred[:, np.newaxis, :]
            return np.sum(sample_weight[:, np.newaxis, np.newaxis] *
                          prob_mat * cost_matrix[np.newaxis, :, :])

    def _logloss_estimation(self, prob_true, prob_pred):
        return -np.sum(prob_true * np.log(prob_pred + np.finfo(float).eps))


class MonteCarloEER(ExpectedErrorReduction):
    """
    Roy McCallum
    """
    def __init__(self, method='misclassification_loss', cost_matrix=None,
                 missing_label=MISSING_LABEL, random_state=None):
        super().__init__(
            enforce_mapping=False,
            cost_matrix=cost_matrix,
            missing_label=missing_label,
            random_state=random_state,
        )
        self.method = method

    def _validate_init_params(self):
        # TODO check if cost_matrix is None for log_loss

        # Validate method.
        if not isinstance(self.method, str):
            raise TypeError('{} is an invalid type for method. Type {} is '
                            'expected'.format(type(self.method), str))
        if self.method not in ['misclassification_loss', 'log_loss']:
            raise ValueError(
                f"Supported methods are `misclassification_loss`, or"
                f"`log_loss` the given one is: {self.method}"
            )

    def _estimate_error_for_candidate(self, uclf, idx_cx, cy, idx_train,
                                      idx_cand, idx_eval, w_eval):
        uclf.fit_add_cand(idx_cx, cy)
        probs = uclf.predict_proba(idx_eval)

        if self.method == 'misclassification_loss':
            preds = np.argmin(np.dot(probs, self.cost_matrix_), axis=1)
            err = self._risk_estimation(probs, preds,
                                        self.cost_matrix_, w_eval[idx_eval])
        elif self.method == 'log_loss':
            err = self._logloss_estimation(probs, probs)
        else:
            raise ValueError(
                f"Supported methods are `misclassification_loss`, or"
                f"`log_loss` the given one is: {self.method}"
            )

        return err

    def _precompute_and_fit_clf(self, uclf, X_full, y_full, idx_train, idx_cand,
                                idx_eval):
        uclf.precompute(idx_train, idx_cand)
        uclf.precompute(idx_train, idx_eval)
        uclf.precompute(idx_cand, idx_eval)
        uclf = super()._precompute_and_fit_clf(
            uclf, X_full, y_full, idx_train, idx_cand, idx_eval
        )
        return uclf


class ValueOfInformationEER(ExpectedErrorReduction):
    """
    Kapour
    Joshi https://ieeexplore.ieee.org/document/6127880
    Margeniantu

    only MCL.
    """
    def __init__(self, cost_matrix=None, consider_unlabeled=True,
                 consider_labeled=True, candidate_to_labeled=True,
                 missing_label=MISSING_LABEL, random_state=None):
        super().__init__(
            enforce_mapping=True,
            cost_matrix=cost_matrix,
            missing_label=missing_label,
            random_state=random_state,
        )
        self.consider_unlabeled = consider_unlabeled
        self.consider_labeled = consider_labeled
        self.candidate_to_labeled = candidate_to_labeled

    def _validate_init_params(self):
        check_type(self.consider_unlabeled, 'consider_unlabeled', bool)
        check_type(self.consider_labeled, 'consider_labeled', bool)
        check_type(self.candidate_to_labeled, 'candidate_to_labeled', bool)

    def query(self, X, y, clf, sample_weight=None,
              fit_clf=True, ignore_partial_fit=True,
              candidates=None, batch_size=1, return_utilities=False):

        # TODO check if candidates are only unlabeled ones if given

        return super().query(X, y, clf, sample_weight=sample_weight,
                             fit_clf=fit_clf,
                             ignore_partial_fit=ignore_partial_fit,
                             candidates=candidates,
                             sample_weight_candidates=None,
                             X_eval=None, sample_weight_eval=None,
                             batch_size=batch_size,
                             return_utilities=return_utilities)

    def _estimate_error_for_candidate(self, uclf, idx_cx, cy, idx_train,
                                      idx_cand, idx_eval, w_eval):
        uclf.fit_add_cand(idx_cx, cy)

        # Handle problem that if only one candidate is remaining, this should
        # be the one to be selected although the error cannot be estimated
        # as there are no instances left for estimating

        le = uclf.get_label_encoder()
        y_eval = uclf.get_y(idx_eval)
        idx_labeled = idx_train[is_labeled(y_eval)]
        y_labeled = uclf.get_y(idx_labeled)
        idx_unlabeled = idx_train[is_unlabeled(y_eval)]

        if self.candidate_to_labeled:
            idx_labeled = np.concatenate([idx_labeled, idx_cx], axis=0)
            y_labeled = np.concatenate([y_labeled, cy], axis=0)
            idx_unlabeled = np.setdiff1d(idx_unlabeled, idx_cx,
                                         assume_unique=True)

        y_labeled_c_id = le.transform(y_labeled)

        err = 0
        if self.consider_labeled:
            if len(idx_labeled) > 0:
                probs = uclf.predict_proba(idx_labeled)
                err += self._risk_estimation(
                    y_labeled_c_id, probs, self.cost_matrix_,
                    w_eval[idx_labeled]
                )

        if self.consider_unlabeled:
            if len(idx_unlabeled) > 0:
                probs = uclf.predict_proba(idx_unlabeled)
                err += self._risk_estimation(
                    probs, probs, self.cost_matrix_, w_eval[idx_unlabeled]
                )

        return err

    def _precompute_and_fit_clf(self, uclf, X_full, y_full, idx_train, idx_cand,
                                idx_eval):
        # for cond_prob
        uclf.precompute(idx_train, idx_cand,
                        fit_params='labeled', pred_params='all')
        # for risk estimation
        if self.consider_labeled:
            uclf.precompute(idx_train, idx_eval,
                            fit_params='labeled', pred_params='labeled')
            uclf.precompute(idx_cand, idx_eval,
                            fit_params='all', pred_params='labeled')
            if self.candidate_to_labeled:
                # idx_train ('labeled'), idx_cand ('all') exists above
                # TODO: consider only equal instances would be sufficient
                uclf.precompute(idx_cand, idx_cand,
                                fit_params='all', pred_params='all')
        if self.consider_unlabeled:
            uclf.precompute(idx_train, idx_eval,
                            fit_params='labeled', pred_params='unlabeled')
            uclf.precompute(idx_cand, idx_eval,
                            fit_params='all', pred_params='unlabeled')

        uclf = super()._precompute_and_fit_clf(
            uclf, X_full, y_full, idx_train, idx_cand, idx_eval
        )
        return uclf


class UpdateIndexClassifier():
    def __init__(self, clf, X, y, sample_weight,
                 fit_clf=True, ignore_partial_fit=True,
                 missing_label=MISSING_LABEL):
        self.clf = clf
        self.X = X
        self.y = y
        self.sample_weight = sample_weight

        self.fit_clf = fit_clf

        self.missing_label = missing_label
        self.missing_label_ = self.missing_label # TODO

        # TODO check fit_clf and ignore_partial_fit

        # use partial fit if applicable
        check_type(ignore_partial_fit, 'ignore_partial_fit', bool)
        partial_fit = hasattr(clf, 'partial_fit') and not ignore_partial_fit

        self.use_partial_fit = \
            hasattr(clf, 'partial_fit') and not ignore_partial_fit

        # precompute PWC
        if isinstance(self.clf, PWC):
            if not hasattr(self, 'clf_'):
                self.clf_ = clone(self.clf)
            else:
                self.clf_ = clone(self.clf_)

            self.pwc_metric_ = self.clf.metric
            self.pwc_metric_dict_ = \
                {} if self.clf.metric_dict is None else self.clf.metric_dict
            self.clf_.metric = 'precomputed'
            self.clf_.metric_dict = {}

            self.pwc_K_ = np.full([len(self.X), len(self.X)], np.nan)

    def get_classes(self):
        return self.clf_.classes_

    def get_X(self, idx):
        return self.X[idx]

    def get_y(self, idx):
        return self.y[idx]

    def get_label_encoder(self):
        return self.clf_._le

    def precompute(self, idx_fit, idx_pred, fit_params='all',
                   pred_params='all'):

        # precompute PWC
        if isinstance(self.clf, PWC):
            if fit_params == 'all':
                idx_fit_ = idx_fit
            elif fit_params == 'labeled':
                idx_fit_ = \
                    idx_fit[is_labeled(
                        self.y[idx_fit], missing_label=self.missing_label_
                    )]
            elif fit_params == 'unlabeled':
                idx_fit_ = \
                    idx_fit[is_unlabeled(
                        self.y[idx_fit], missing_label=self.missing_label_
                    )]
            else:
                raise ValueError(f'`fit_params`== {fit_params} not defined')

            if pred_params == 'all':
                idx_pred_ = idx_pred
            elif pred_params == 'labeled':
                idx_pred_ = \
                    idx_pred[is_labeled(
                        self.y[idx_pred], missing_label=self.missing_label_
                    )]
            elif pred_params == 'unlabeled':
                idx_pred_ = \
                    idx_pred[is_unlabeled(
                        self.y[idx_pred], missing_label=self.missing_label_
                    )]
            else:
                raise ValueError(f'`pred_params`== {pred_params} not defined')

            if len(idx_fit_) > 0 and len(idx_pred_) > 0:
                self.pwc_K_[np.ix_(idx_fit_, idx_pred_)] = \
                    pairwise_kernels(self.X[idx_fit_], self.X[idx_pred_],
                                     self.pwc_metric_, **self.pwc_metric_dict_)


    def fit(self, idx, set_train_idx=True):
        if set_train_idx:
            self.train_idx_ = idx
        self.idx_ = idx

        if not hasattr(self, 'clf_'):
            self.clf_ = self.clf

        if self.fit_clf:
            self.clf_ = clone(self.clf_)
            if self.sample_weight is None:
                self.clf_.fit(self.X[idx], self.y[idx])
            else:
                self.clf_.fit(self.X[idx], self.y[idx],
                              self.sample_weight[idx])
        else:
            check_is_fitted(self.clf_)
            self.clf_ = deepcopy(self.clf_)

        return self.clf_

    def fit_add_cand(self, c_idx, y):

        if self.use_partial_fit:
            if self.sample_weight is None:
                self.clf_ = deepcopy(self.clf).partial_fit(self.X[c_idx], y)
            else:
                self.clf_ = deepcopy(self.clf).partial_fit(
                    self.X[c_idx], y, self.sample_weight[c_idx]
                )
        else:
            new_idx = np.concatenate([self.train_idx_, c_idx], axis=0)
            y_new = np.concatenate([self.y[self.train_idx_], y], axis=0)
            if self.sample_weight is None:
                self.clf_ = clone(self.clf_).fit(self.X[new_idx], y_new)
                self.idx_ = new_idx
            else:
                self.clf_ = clone(self.clf_).fit(self.X[new_idx], y_new,
                                              self.sample_weight[new_idx])
                self.idx_ = new_idx

    def predict_proba(self, idx):

        if hasattr(self, 'pwc_K_'):
            # check if results contain NAN
            res = self.clf_.predict_proba(self.pwc_K_[self.idx_, :][:, idx].T)
            if np.isnan(res).any():
                raise ValueError('Error in defining what should be '
                                 'pre-computed. Not all necessary information '
                                 'is available which results in NaNs in '
                                 '`predict_proba`.')
            return res
        else:
            return self.clf_.predict_proba(self.X[idx])
