import math
import typing

import numpy as np
from sklearn.cluster import KMeans

import scipy.stats as sps

from river import utils


class MicroCluster:
    features: typing.Dict = {}

    def __init__(
            self,
            x=np.random.rand(100, 5),
            y=None,
            time_stamp=1
    ):
        self.features: typing.Dict = {
            "ls_x": np.sum(x, 0),
            "ls_t": time_stamp,
            "ss_t": np.square(time_stamp),
            "n": len(x),
            "M": np.square(x[0] - np.divide(x[0], len(x))),
        }
        self.x = x

        self.labeled_samples = np.empty((0,), dtype=object)

        if (y is not None) and (not np.isnan(y)):
            self.labeled_samples = np.array((x[0], y), dtype=object)

    @property
    def center(self):
        return np.divide(self.features["ls_x"], self.features["n"])

    @property
    def mean(self):
        return self.features["ls_x"] / self.features["n"]

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
        self.features["ss_t"] += np.square(t)

        self.features["n"] += 1

        self.features["M"] += (x - past_mean) * (x - (self.features["ls_x"] / self.features["n"]))

        if not np.isnan(y):
            if len(self.labeled_samples) == 0:
                self.labeled_samples = np.array((x, y), dtype=object)
            else:
                self.labeled_samples = np.vstack([self.labeled_samples, (x, y)])

        self.x = np.vstack([self.x, x[np.newaxis, ...]])

    def __iadd__(self, other: "MicroCluster"):
        addterm_m = self.features["n"] * other.features["n"] / (self.features["n"] + other.features["n"])
        addterm_m *= np.square(self.mean - other.mean)
        
        self.features = {k: self.features[k] + other.features[k] for k, value in other.features.items()}
        self.features["M"] += addterm_m
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
            time_window=1000
    ):
        self.mc = micro_cluster

        self.n_micro_clusters = n_micro_clusters
        self.r_factor = r_factor
        self.seed = seed

        self.centers: dict[int, []] = {}
        self.micro_clusters: dict[int, micro_cluster] = {}

        self._timestamp = -1
        self.time_window = time_window
        self._initialized = False

        self._kmeans_mc = KMeans(n_clusters=n_micro_clusters, n_init=n_init_train)

        self.n_init_train = n_init_train

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

    def fit_one(self, X, y=None):
        if not self._initialized:
            sample = np.array((X, y), dtype=object)
            if len(self.init_train) == 0:
                self.init_train = sample
                return X

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
            self.cluster_test[nearest_mc_id] = np.vstack([self.cluster_test[nearest_mc_id], (X, y)])
            return nearest_mc_id

        # Else Merge or delete Cluster
        nearest_mc_id = self._update_clusters((X,y))
        return nearest_mc_id

    def n_sum(self):
        ns = [mc.features["n"] for i, mc in self.micro_clusters.items()]
        return sum(ns)

    def _init_clusters(self):
        x_init = np.vstack(self.init_train[:, 0])
        y_init = self.init_train[:, 1]
        self._kmeans_mc.fit(x_init)
        self.centers = {i: X for i, X in enumerate(self._kmeans_mc.cluster_centers_)}
        self.micro_clusters = {i: self.mc(x=X[np.newaxis, ...], time_stamp=self.n_micro_clusters - 1) for i, X in
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
            self.micro_clusters[del_id] = MicroCluster(X[np.newaxis, ...], y, self._timestamp)
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
        self.micro_clusters[closest_a].x = np.vstack([self.micro_clusters[closest_a].x, self.micro_clusters[closest_b].x])

        self.micro_clusters[closest_b] = MicroCluster(X[np.newaxis, ...], y, self._timestamp)

        return closest_b

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
            stds[i] = np.std(mc.x, axis=0)
            stds_m[i] = np.sqrt(mc.features["M"] / (mc.features["n"]))

        return stds, stds_m

    @property
    def initialized(self):
        return self._initialized