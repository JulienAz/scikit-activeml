from collections import deque

import numpy as np

from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import SklearnClassifier
from skactiveml.utils import MISSING_LABEL


class XuPairedEnsembleClassifier(SkactivemlClassifier):

    def __init__(
            self,
            clf_type,
            labeling_strategy,
            classes=None,
            missing_label=MISSING_LABEL,
            cost_matrix=None,
            random_state=None,
            w=300,
            detection_threshold=0.2,
            classifier_param_dict=None,
    ):
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )
        self.clf_type = clf_type
        self.stable_clf = SklearnClassifier(clf_type(), **classifier_param_dict)
        self.reactive_clf = SklearnClassifier(clf_type(), **classifier_param_dict)

        self.classifier_param_dict = classifier_param_dict

        self.labeling_strategy = labeling_strategy
        self.w = w
        self.detection_threshold = detection_threshold * w

        self.classes = classes

        self.change_state = deque(maxlen=w)
        self.training_window_X = deque(maxlen=w)
        self.training_window_y = deque(maxlen=w)

        self.random_state = random_state

    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        self.reactive_clf.fit(X, y, sample_weight=sample_weight, **fit_kwargs)
        return self.stable_clf.fit(X, y, sample_weight=sample_weight, **fit_kwargs)

    def partial_fit(self, X, y, logger=None, sample_weight=None, **fit_kwargs):
        if y[0] is not self.stable_clf.missing_label:
            # Update changestate and adjust structure if necessary
            change_detected = self.update_change_state(X, y)
            if change_detected:
                self.swap_classifier()

            #Train stable classifier
            self.stable_clf.partial_fit(X.reshape([1, -1]), np.array([y]))

            #If labeled by random strategy train reactive classifier
            if self.labeling_strategy.random_sampled:
                self.training_window_X.append(X[0])
                self.training_window_y.append(y[0])
                #self.reactive_clf.reset()
                #self.reactive_clf.fit(self.training_window_X, np.array(self.training_window_y))

                self.reactive_clf.partial_fit(X.reshape([1, -1]), np.array([y]))

    def update_change_state(self, X, y):
        y_stable = self.stable_clf.predict(X)
        y_reactive = self.reactive_clf.predict(X)

        if (y_stable != y) & (y_reactive == y):
            self.change_state.append(1)
        else:
            self.change_state.append(0)

        if sum(self.change_state) > self.detection_threshold * len(self.change_state):
            return True
        return False

    def swap_classifier(self):
        self.stable_clf = self.reactive_clf
        self.change_state = deque(maxlen=self.w)
        self.reactive_clf = SklearnClassifier(self.clf_type(), **self.classifier_param_dict)
        self.reactive_clf.fit(self.training_window_X, np.array(self.training_window_y))

    def predict_proba(self, X):
        return self.stable_clf.predict_proba(X)

    def predict(self, X):
        return self.stable_clf.predict(X)
