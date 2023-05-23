import numpy as np
from sklearn.neighbors import KernelDensity

from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.stream.clustering import CluStream
from skactiveml.utils import MISSING_LABEL


class CluStreamClassifier(SkactivemlClassifier):

    def __init__(
        self,
        estimator_clf,
        clustering=CluStream(),
        freq_pred_clf=None,
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        random_state=None,
        metric_dict=None
    ):
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )
        self.metric_dict = metric_dict
        self.estimator_clf = estimator_clf
        self.freq_pred_clf = freq_pred_clf
        self.clustering = clustering

    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        for t, (x_t, y_t) in enumerate(zip(X, y)):
            self.clustering.fit_one(x_t, y_t)
        return self.estimator_clf.fit(X, y, sample_weight=None, **fit_kwargs)

    def partial_fit(self, X, y, sample_weight=None, **fit_kwargs):
        self.clustering.fit_one(X[0], y[0])
        if y[0] is not self.estimator_clf.missing_label:
            self.estimator_clf.partial_fit(X.reshape([1, -1]), np.array([y]), **fit_kwargs)
        #return self.estimator_clf.partial_fit(X, y, sample_weight)

    def predict_proba(self, X):
        return self.estimator_clf.predict_proba(X)

    def predict(self, X):
        return self.estimator_clf.predict(X)

    def predict_freq(self, X, logger=None):
        if self.clustering.initialized:
            cluster_id, _ = self.clustering.nearest_cluster(X)
            mc = self.clustering.micro_clusters[cluster_id]

            if not mc.labeled_samples.size == 0:
                if mc.labeled_samples.ndim > 1:
                    X_fit = np.stack(mc.labeled_samples[:, 0])
                    y_fit = np.array(mc.labeled_samples[:, 1], dtype=float)
                else:
                    X_fit = [mc.labeled_samples[0]]
                    y_fit = np.array([mc.labeled_samples[1]])
                pwc = ParzenWindowClassifier(
                    metric="rbf",
                    metric_dict=self.metric_dict,
                    missing_label=self.estimator_clf.missing_label,
                    classes=self.estimator_clf.classes,
                )

                pwc.fit(X=X_fit, y=y_fit)
                n = pwc.predict_freq(X).sum(axis=1, keepdims=True)
                if logger is not None:
                    logger.track_lbl_frequency(n.sum(axis=1, keepdims=True)[0][0])
                    logger.track_cluster(cluster_id)
                pred_proba = self.estimator_clf.predict_proba(X)
                k_vec = n * pred_proba

                kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(mc.x)
                # X_cluster_density = np.array([kde.score(X)]) TODO: Density seems to be broken, high negative values
                X_cluster_density = 1
                return k_vec * X_cluster_density

        if logger is not None:
            logger.track_lbl_frequency(1)
        return self.estimator_clf.predict_proba(X)



