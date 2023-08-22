import numpy as np
from sklearn.neighbors import KernelDensity

from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.stream.clustering import CluStream
from skactiveml.stream.clustering._clu_stream_al import MicroClfCluster
from skactiveml.utils import MISSING_LABEL


class CluStreamClassifier(SkactivemlClassifier):

    def __init__(
            self,
            clf_type,
            freq_pred_clf=None,
            classes=None,
            missing_label=MISSING_LABEL,
            cost_matrix=None,
            random_state=None,
            metric_dict=None,
            refit=False,
            clustering_param_dict=None,
            classifier_param_dict=None,
            change_detector_type='prediction_error'
    ):
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )
        self.metric_dict = metric_dict

        # Init Classifier        classifier_param_dict['']
        self.estimator_clf = SklearnClassifier(clf_type(), **classifier_param_dict)
        self.freq_pred_clf = freq_pred_clf

        self.clustering = CluStream(**clustering_param_dict)

        self.change_detector_type = change_detector_type

        self.random_state = random_state

        self.refit = refit

        self.classes = classes

    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        if self.clustering.initialized:
            self.clustering.fit_one(X[-1], y[-1])
        else:
            for t, (x_t, y_t) in enumerate(zip(X, y)):
                self.clustering.fit_one(x_t, y_t)
        return self.estimator_clf.fit(X, y, sample_weight=sample_weight, **fit_kwargs)

    def partial_fit(self, X, y, acc_logger=None, statistic_logger=None, detection_logger=None, sample_weight=None, **fit_kwargs):
        mc_id_fitted, mc_id_merged = self.clustering.fit_one(X[0], y[0])

        if not y == self.missing_label:
            self.update_change_detector(X, y, mc_id_fitted, mc_id_merged)

        if acc_logger is not None:
            acc_logger.track_cluster(mc_id_fitted)

        # If is refit approach #Todo: Cluster adaption should be in Clustering itself. Once decided for approach implementation must be refactored
        if self.refit:
            changed_clusters = []
            change_detections = [False] * self.clustering.n_micro_clusters
            for mc_id, mc in self.clustering.micro_clusters.items():
                change_detections[mc_id] = mc.change_detector.drift_detected
                if mc.change_detector.drift_detected:
                    changed_clusters.append(mc_id)
                change_detections[mc_id] = mc.change_detector.drift_detected
            if not statistic_logger == None:
                statistic_logger.track_change_detection(change_detections)

            if detection_logger is not None:
                detection_logger.track_change_detection(any(change_detections))
            # If change_detector of cluster is positiv, corresponding cluster is cleared
            # self.clustering.clear_cluster(mc_id)

            # Refitting Classifier on Labeled Samples in all Cluster
            if changed_clusters:
                for mc_id in changed_clusters:
                    self.clustering.clear_cluster(mc_id)
                self.refit_on_cluster(X, y, sample_weight=sample_weight, **fit_kwargs)

        if y[0] is not self.estimator_clf.missing_label:
            return self.estimator_clf.partial_fit(X.reshape([1, -1]), np.array([y]))

    def refit_on_cluster(self, X, y, sample_weight=None, **fit_kwargs):
        # labeled_data = np.array([np.array(mc.labeled_samples) for mc_id, mc in self.clustering.micro_clusters.items()])
        X = []
        y = []

        X_test = []
        y_test = []
        # Collecting the Labeled Samples from the clusters
        for mc_id, mc in self.clustering.micro_clusters.items():
            # Hard coded for rbf_generator change adaption !!!
            # if not len(mc.labeled_samples) == 0:
            #    count_2 = np.count_nonzero(mc.labeled_samples[:, 1] == 2)
            #    is_majority_two = count_2 > mc.labeled_samples[:,1].size / 2#

            #    if is_majority_two:
            #        features, targets = zip(*mc.labeled_samples)
            #        X.extend(features)
            #        y.extend(targets)
            #    else:
            #        self.clustering.clear_cluster(mc_id)
            if not len(mc.labeled_samples[0]) == 0:
                X.extend(mc.labeled_samples[0])
                y.extend(mc.labeled_samples[1])

            if not len(mc.test) == 0:
                features, targets = zip(*mc.test)
                X_test.extend(features)
                y_test.extend(targets)

            #assert np.array_equal(X_test, X)
            #assert np.array_equal(y_test, y)
        # Convert the lists to NumPy arrays
        if not len(X) == 0:
            X = np.vstack(X)
            y = np.array(y)
            return self.estimator_clf.fit(X, y, sample_weight=sample_weight, **fit_kwargs)

    def fit_window(self, X, y, sample_weight=None, **fit_kwargs):
        self.clustering.fit_one(X[-1], y[-1])
        return self.estimator_clf.fit(X, y, sample_weight=sample_weight, **fit_kwargs)

    def predict_proba(self, X):
        return self.estimator_clf.predict_proba(X)

    def predict(self, X):
        return self.estimator_clf.predict(X)

    #!!!Depricated
    def predict_freq(self, X, logger=None):
        if self.clustering.initialized & len(self.clustering.micro_clusters) != 0:
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
                pred_proba = self.predict_proba(X)
                k_vec = n * pred_proba

                # kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(mc.x)
                # X_cluster_density = np.array([kde.score(X)]) TODO: Density seems to be broken, high negative values
                X_cluster_density = 1
                return k_vec * X_cluster_density

        if logger is not None:
            logger.track_lbl_frequency(1)
        return self.estimator_clf.predict_proba(X)

    def update_change_detector(self, X, y, mc_id_fitted, mc_id_merged):
        if self.change_detector_type == 'prediction_error':
            prediction = self.predict(X)
            self.clustering.micro_clusters[mc_id_fitted].change_detector.update(not prediction == y)
            return

        if self.change_detector_type == 'entropy':
            cluster_entropy = self.clustering.micro_clusters[mc_id_fitted].class_entropy
            self.clustering.micro_clusters[mc_id_fitted].change_detector.update(cluster_entropy)

            if mc_id_merged is not None:
                self.clustering.micro_clusters[mc_id_merged].reset_change_detector()
                cluster_entropy = self.clustering.micro_clusters[mc_id_merged].class_entropy
                self.clustering.micro_clusters[mc_id_merged].change_detector.update(cluster_entropy)
            return


class CluStreamEnsembleClassifier(CluStreamClassifier):
    def predict_proba(self, X):
        # Get Clf of cluster the point is assigned to
        cluster_id, _ = self.clustering.nearest_cluster(X)

        # Get weighted probabilities of base estimator and cluster estimator
        if (cluster_id in self.clustering.micro_clusters):
            if (len(self.clustering.micro_clusters[cluster_id].labeled_samples[0]) > 0):
                mc_clf = self.clustering.micro_clusters[cluster_id].clf
                # Get weighted probabilities of base estimator and cluster estimator
                cluster_proba = self.clustering.micro_clusters[cluster_id].predict_proba()

                proba = self.estimator_clf.predict_proba(X) * 0.5 + cluster_proba * 0.5
                #proba = self.estimator_clf.predict_proba(X) * 0.5 + mc_clf.predict_proba(X) * 0.5

            else:
                proba = self.estimator_clf.predict_proba(X)
        else:
            proba = self.estimator_clf.predict_proba(X)
        return proba

    def partial_fit(self, X, y, acc_logger=None, statistic_logger=None, detection_logger=None, sample_weight=None, **fit_kwargs):
        mc_id_fitted, mc_id_merged = self.clustering.fit_one(X[0], y[0])

        if not y == self.missing_label:
            self.update_change_detector(X, y, mc_id_fitted, mc_id_merged)

        if acc_logger is not None:
            acc_logger.track_cluster(mc_id_fitted)

        # If is refit approach #Todo: Cluster adaption should be in Clustering itself. Once decided for approach implementation must be refactored
        if self.refit:
            changed_clusters = []
            change_detections = [False] * self.clustering.n_micro_clusters
            for mc_id, mc in self.clustering.micro_clusters.items():
                change_detections[mc_id] = mc.change_detector.drift_detected
                if mc.change_detector.drift_detected:
                    changed_clusters.append(mc_id)
                change_detections[mc_id] = mc.change_detector.drift_detected
            if not statistic_logger == None:
                statistic_logger.track_change_detection(change_detections)
            # If change_detector of cluster is positiv, corresponding cluster is cleared
            # self.clustering.clear_cluster(mc_id)

            # Refitting Classifier on Labeled Samples in all Cluster
            if changed_clusters:
                for mc_id in changed_clusters:
                    self.clustering.clear_cluster(mc_id)
                self.refit_on_cluster(X, y, sample_weight=sample_weight, **fit_kwargs)

            if detection_logger is not None:
                detection_logger.track_change_detection(any(change_detections))

        if y[0] is not self.estimator_clf.missing_label:
            return self.estimator_clf.partial_fit(X.reshape([1, -1]), np.array([y]))

    def predict(self,
                X):  # !!! TODO: figure out wheather only prediction of cluster clf for change detection or weighted
        proba = self.predict_proba(X)
        return [np.argmax(proba)]
