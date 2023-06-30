import math
import typing

import numpy as np
from numpy import float64
from river.drift import ADWIN
from river.drift.binary import DDM
from scipy.stats import entropy
from sklearn.cluster import KMeans

import scipy.stats as sps

from river import utils


class MicroCluster:
    features: typing.Dict = {}

    def __init__(
            self,
            x=np.random.rand(100, 5),
            y=None,
            time_stamp=1,
            n_classes=None,
            change_detector=DDM
    ):
        self.features: typing.Dict = {
            "ls_x": np.sum(x, 0),
            "ls_t": time_stamp,
            "ss_t": float64(np.square(time_stamp)),
            "n": len(x),
            "class_dist": np.zeros(n_classes),
            "M": np.square(x[0] - np.divide(x[0], len(x))),
        }
        #self.x = x

        self.n_classes = n_classes
        self.labeled_samples = np.empty((0,), dtype=object)

        # Adwin params
        #self.change_detector = change_detector(clock=1, delta=0.1)

        # DDM params
        self.change_detector = change_detector(warm_start=5, drift_threshold=1.5)

        if (y is not None) and (not np.isnan(y)):
            self.features["class_dist"][y] += 1
            self.labeled_samples = np.array((x[0], y), dtype=object).reshape(1, 2)
            self.change_detector.update(self.class_entropy)

    @property
    def center(self):
        return np.divide(self.features["ls_x"], self.features["n"])

    @property
    def mean(self):
        return self.features["ls_x"] / self.features["n"]

    @property
    def class_entropy(self):
        if np.sum(self.features['class_dist']) == 0:
            return 0

        class_probabilities = self.features['class_dist'] / len(self.labeled_samples)
        return entropy(class_probabilities, base=self.n_classes)

    def radius(self):
        std = np.sqrt(self.features["M"] / (self.features["n"]))
        std_sum = sum(std) / len(std)

        return std_sum

    def relevance_stamp(self):
        if self.features["n"] < 2 * 100:
            return self.features["ls_t"] / self.features["n"]

        mean = self.features["ls_t"] / self.features["n"]

        sigma = sum(np.sqrt(self.features["M"] / self.features["n"])) / len(self.features["M"])
        percentile = 100 / (2 * self.features["n"])

        t_dist = sps.norm(loc=mean, scale=sigma)
        return t_dist.ppf(percentile)

    def add(self, data):
        (x, y),  t = data

        past_mean = self.features["ls_x"] / self.features["n"]

        self.features["ls_x"] += x

        self.features["ls_t"] += t
        self.features["ss_t"] += float64(np.square(t))

        self.features["n"] += 1

        self.features["M"] += (x - past_mean) * (x - (self.features["ls_x"] / self.features["n"]))

        if y is not None:   ####TODO: Missing Label hinzufügen
            self.features["class_dist"][y] += 1
            if len(self.labeled_samples) == 0:
                self.labeled_samples = np.array([x, y], dtype=object).reshape(1, 2)
            else:
                self.labeled_samples = np.vstack([self.labeled_samples, np.array((x, y), dtype=object)])
            self.change_detector.update(self.class_entropy)

        #self.x = np.vstack([self.x, x[np.newaxis, ...]])

    def __iadd__(self, other: "MicroCluster"):
        addterm_m = self.features["n"] * other.features["n"] / (self.features["n"] + other.features["n"])
        addterm_m *= np.square(self.mean - other.mean)
        
        self.features = {k: self.features[k] + other.features[k] for k, value in other.features.items()}
        self.features["M"] += addterm_m

        if len(other.labeled_samples) > 0:
            self.labeled_samples = np.vstack([self.labeled_samples.reshape(-1, 2), other.labeled_samples.reshape(-1, 2)])
            self.change_detector.update(self.class_entropy)

        return self


class CluStream:
    init_train = []
    n_init_train: int = -1
    cluster_test = []

    def __init__(
            self,
            n_micro_clusters: int = 10,
            seed: int = None,
            n_init_train: int = 100,
            init_train=None,
            r_factor=3,
            micro_cluster=MicroCluster,
            time_window=1000,
            random_state=None,
            n_classes=None
    ):
        self.mc = micro_cluster

        self.n_micro_clusters = n_micro_clusters
        self.r_factor = r_factor
        self.seed = seed
        self.n_classes = n_classes

        self.centers: dict[int, []] = {}
        self.micro_clusters: dict[int, micro_cluster] = {}

        self._timestamp = -1
        self.time_window = time_window
        self._initialized = False

        self._kmeans_mc = KMeans(n_clusters=n_micro_clusters, n_init=n_init_train, random_state=random_state)

        self.n_init_train = n_init_train

        self.free_cluster = []

        if init_train is not None:
            self.init_train = init_train
            self._init_clusters()

    def reset(self, init_train=None):
        self.centers: dict[int, []] = {}
        micro_cluster = self.mc
        self.micro_clusters: dict[int, micro_cluster] = {}

        self._initialized = False

        self._kmeans_mc = KMeans(n_clusters=self.n_micro_clusters, n_init=self.n_init_train)

        if init_train is not None:
            self.init_train = self.init_train
            self._init_clusters()
        else:
            self.init_train = []

    def fit_one(self, X, y=None): #TODO: Fit() for multiple samples
        if not self._initialized:
            sample = np.array((X, y), dtype=object)
            if len(self.init_train) == 0:
                self.init_train = sample
                return X, False

            self.init_train = np.vstack((self.init_train, sample))
            if len(self.init_train) == self.n_init_train:
                self._init_clusters()

            return X

        self._timestamp += 1
        nearest_mc_id, distance = self.nearest_cluster(X)
        nearest_cluster = self.micro_clusters[nearest_mc_id]

        # Check if Point fits in cluster (distance < r * radius)
        # If cluster only containts 1 point, radius is distance to next cluster
        if nearest_cluster.features["n"] == 1:
            radius = math.inf
            for mc_id, mc in self.micro_clusters.items():
                if mc_id == nearest_mc_id:
                    continue
                current_distance = self._distance(mc.center, nearest_cluster.center)
                radius = min(current_distance, radius)
        else:
            radius = nearest_cluster.radius()

        # If point fits in cluster -> add and return
        if distance < self.r_factor * radius:
            self.micro_clusters[nearest_mc_id].add(((X, y), self._timestamp))
            self.cluster_test[nearest_mc_id] = np.vstack([self.cluster_test[nearest_mc_id], np.array((X, y), dtype=object)])
            return nearest_mc_id, self.micro_clusters[nearest_mc_id].change_detector.drift_detected

        # Else check if free clusters are available
        if self.free_cluster:
            free_cluster_id = self.free_cluster.pop(0)
            self.micro_clusters[free_cluster_id] = self.mc(X[np.newaxis, ...], y, self._timestamp, self.n_classes)
            self.cluster_test[free_cluster_id] = np.array((X, y), dtype=object) #!!! For cluster analysis

        # Else Merge or delete Cluster
        nearest_mc_id = self._update_clusters((X,y))
        return nearest_mc_id, self.micro_clusters[nearest_mc_id].change_detector.drift_detected

    def n_sum(self):
        ns = [mc.features["n"] for i, mc in self.micro_clusters.items()]
        return sum(ns)

    def _init_clusters(self):
        x_init = np.vstack(self.init_train[:, 0])
        y_init = self.init_train[:, 1]
        self._kmeans_mc.fit(x_init)
        self.centers = {i: X for i, X in enumerate(self._kmeans_mc.cluster_centers_)}
        self.micro_clusters = {i: self.mc(x=X[np.newaxis, ...],
                                          time_stamp=self.n_micro_clusters - 1,
                                          n_classes=self.n_classes)
                               for i, X in
                               self.centers.items()}
        self.cluster_test = [np.array((X, np.nan), dtype=object) for X in self._kmeans_mc.cluster_centers_]
        self._initialized = True

        for x, y in self.init_train:
            self.fit_one(x, y)

    def _update_clusters(self, sample):

        (X, y) = sample
        # Check if a Cluster is old enough to be deleted
        threshold = self._timestamp - self.time_window

        # Delete old micro-cluster if its relevance stamp is smaller than the threshold
        del_id = None
        for i, mc in self.micro_clusters.items():
            if mc.relevance_stamp() < threshold:
                del_id = i
                break

        if del_id is not None:
            self.micro_clusters[del_id] = MicroCluster(X[np.newaxis, ...], y, self._timestamp, self.n_classes)
            self.cluster_test[del_id] = np.array((X, y), dtype=object)
            return del_id

        # Else merge the two closest clusters
        closest_a = 0
        closest_b = 0
        min_distance = math.inf
        for i, mc_a in self.micro_clusters.items():
            for j, mc_b in self.micro_clusters.items():
                if i <= j:
                    continue
                dist = self._distance(mc_a.center, mc_b.center)
                if dist < min_distance:
                    min_distance = dist
                    closest_a = i
                    closest_b = j

        self.cluster_test[closest_a] = np.vstack([self.cluster_test[closest_a], self.cluster_test[closest_b]])
        self.cluster_test[closest_b] = np.array((X, y), dtype=object)

        self.micro_clusters[closest_a] += self.micro_clusters[closest_b]
        #self.micro_clusters[closest_a].x = np.vstack([self.micro_clusters[closest_a].x, self.micro_clusters[closest_b].x])

        self.micro_clusters[closest_b] = MicroCluster(X[np.newaxis, ...], y, self._timestamp, self.n_classes)

        return closest_b

    # Invoked when change in cluster is detected
    def clear_cluster(self, cluster_id):
        # !!! Old implementation, cluster is simply reinitialized with its old centroid
        #centroid = self.micro_clusters[cluster_id].center
        #self.micro_clusters[cluster_id] = self.mc(centroid[np.newaxis, ...], None, self._timestamp, self.n_classes)
        #self.cluster_test[cluster_id] = np.array((centroid, np.nan), dtype=object)

        # !!! New implementation to test: Cluster is added to free cluster list
        # Delete corresponding microcluster from dictionary
        del self.micro_clusters[cluster_id]

        # Add clusterindice to free cluster list
        self.free_cluster.append(cluster_id)

        test = self.micro_clusters

    def nearest_cluster(self, X):
        closest_distance = math.inf
        closest_mc_idx = -1

        for idx, cluster in self.micro_clusters.items():
            dist = self._distance(cluster.center, X)
            if dist < closest_distance:
                closest_distance = dist
                closest_mc_idx = idx

        return closest_mc_idx, closest_distance

    def _distance(self, point_a, point_b):
        return np.linalg.norm(point_a - point_b)

    def stds(self):
        stds = np.ones((self.n_micro_clusters, 2))
        stds_m = np.ones((self.n_micro_clusters, 2))

        for i, mc in self.micro_clusters.items():
            #stds[i] = np.std(mc.x, axis=0)
            stds_m[i] = np.sqrt(mc.features["M"] / (mc.features["n"]))

        return stds, stds_m

    @property
    def initialized(self):
        return self._initialized