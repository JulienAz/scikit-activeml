import numpy as np
from skmultiflow.drift_detection import DDM

from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import SklearnClassifier
from skactiveml.utils import MISSING_LABEL


class ZliobateClassifier(SkactivemlClassifier):

    def __init__(
        self,
        clf_type,
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        random_state=None,
        metric_dict=None,
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
        self.classifier_param_dict = classifier_param_dict

        self.classes=classes

        self.change_clf = None
        self.change_state = None
        self.change_detector = DDM()

        #self.metric_dict = metric_dict

        self.random_state = random_state

        # For plots
        self.instances = []
        self.labels = []
    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        return self.stable_clf.fit(X, y, sample_weight=sample_weight, **fit_kwargs)

    def partial_fit(self, X, y, detection_logger=None):
        if y[0] is not self.stable_clf.missing_label:
            self.stable_clf.partial_fit(X.reshape([1, -1]), np.array([y]))
            self.instances.append(X[0])
            self.labels.append(y[0])
            self._update_change_state(X, y, detection_logger)


    def _update_change_state(self, X, y, detection_looger):
        correct_prediction = self.stable_clf.predict(X) == y
        self.change_detector.add_element(not correct_prediction)

        if self.change_detector.detected_warning_zone():
            if self.change_clf is None:
                self.change_clf = SklearnClassifier(self.clf_type(), **self.classifier_param_dict)

        if self.change_clf is not None:
            self.change_clf.partial_fit(X.reshape([1, -1]), np.array([y]))

        if self.change_detector.detected_change():
            self.instances = [X[0]]
            self.labels = [y[0]]
            if self.change_clf is not None:
                self.stable_clf = self.change_clf
            else:
                self.stable_clf = SklearnClassifier(self.clf_type(), **self.classifier_param_dict)
                self.stable_clf.partial_fit(X.reshape([1, -1]), np.array([y]))

            if detection_looger is not None:
                detection_looger.track_change_detection(True)
        else:
            if detection_looger is not None:
                detection_looger.track_change_detection(False)

            self.change_clf = None

    def predict_proba(self, X):
        return self.stable_clf.predict_proba(X)

    def predict(self, X):
        return self.stable_clf.predict(X)