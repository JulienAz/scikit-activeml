import collections
import random
from datetime import time

import numpy as np
import pandas as pd
import seaborn as sb
import sklearn
from matplotlib import pyplot as plt, patches
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_kernels
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KernelDensity

from skmultiflow.data import HyperplaneGenerator

from skactiveml.classifier._clu_stream_classifier import CluStreamClassifier
from skactiveml.utils import call_func
from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.stream import StreamProbabilisticAL, StreamRandomSampling, VariableUncertainty
from skactiveml.stream.budgetmanager import BalancedIncrementalQuantileFilter
from skactiveml.stream.clustering._clu_stream_al import MicroCluster, CluStream

REP = "rep"
TIMESTEP = "round"
APPROACH = "Approach"
ACCURACY = "Accuracy"
CLUSTER = "Cluster"
CLASSIFIER = "Classifier"
TIME = "time"
BUDGET = "Budget"
X1 = "X1"
X2 = "X2"
Y = "Label"
GT = "GT_Label"

# Add column names to list
all_ids = [
    TIMESTEP,
    ACCURACY,
    CLUSTER,
    CLASSIFIER,
    X1,
    X2,
    Y,
    GT
]


class ExperimentLogger:

    # Nothing to do here
    def __init__(self):
        self._columns = all_ids
        self._data = []
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._column_indices = {key: i for (i, key) in enumerate(self._columns)}

    # Add own functions for tracking different metrics
    def track_rep(self, value: int):
        self._track_value(value, REP)

    def track_classifier(self, value: str):
        self._track_value(value, CLASSIFIER)

    def track_accuracy(self, value: float):
        self._track_value(value, ACCURACY)

    def track_timestep(self, value: int):
        self._track_value(value, TIMESTEP)

    def track_budget(self, value: int):
        self._track_value(value, BUDGET)

    def track_cluster(self, value: int):
        self._track_value(value, CLUSTER)

    def track_x1(self, value):
        self._track_value(value, X1)

    def track_x2(self, value):
        self._track_value(value, X2)

    def track_y(self, value):
        self._track_value(value, Y)

    def track_gt(self, value):
        self._track_value(value, GT)

    def finalize_round(self):
        self._data.append(self._current_row)
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._data, columns=self._columns)
        self._reset()
        return df

    # Noting to do here
    def track_time(self):
        current_time = time.perf_counter_ns()
        self._track_value(current_time, "time")

    def _track_value(self, newval, id):
        self._current_row[self._index_of(id)] = newval

    def _index_of(self, id):
        return self._column_indices[id]

    def _reset(self):
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._data = []

    @property
    def data(self):
        return self._data


def save_image(filename):
    # PdfPages is a wrapper around pdf
    # file so there is no clash and
    # create files with no error.
    p = PdfPages(filename)

    # get_fignums Return list of existing
    # figure numbers
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]

    # iterating over the numbers in list
    for fig in figs:
        # and saving the files
        fig.savefig(p, format='pdf')

        # close the object
    p.close()


def get_randomseed(random_state):
    return random_state.randint(2 ** 31 - 1)


def next_sample(data, n=1):
    # Randomly choose datapoint from dataset
    X, y = random.sample(data, n)[0]

    # Add noise to features
    X_new = X + np.random.normal(0, 0.1, len(X))
    return X_new, y


def label_sample(X_cand, y_cand, query_strategy, classifier, X=None, y=None):
    X_cand = X_cand.reshape([1, -1])
    y_cand = np.array([[y_cand]])

    # Check Label Strategy decision
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X)
    X_candidate_density = np.array([kde.score(X_cand)])

    sampled_indices, utilities = query_strategy.query(candidates=X_cand,
                                                      return_utilities=True,
                                                      clf=classifier,
                                                      X=np.array(X),
                                                      y=np.array(y),
                                                      utility_weight=X_candidate_density,
                                                      fit_clf=fit_clf
                                                      )

    # create budget_manager_param_dict for BalancedIncrementalQuantileFilter used by StreamProbabilisticAL
    budget_manager_param_dict = {"utilities": utilities}

    # update the query strategy and budget_manager to calculate the right budget
    query_strategy.update(candidates=X_cand,
                          queried_indices=sampled_indices,
                          budget_manager_param_dict=budget_manager_param_dict
                          )

    if len(sampled_indices) > 0:
        return y_cand[0][0]
    else:
        return classifier.missing_label


def find_label(stream, X):
    for (X_stream, y) in stream:
        if np.array_equal(X, X_stream):
            return y


# Test Starting

random_number = 1
n_init_training = 150
stream_length = 10000
n_clusters = 10
n_features = 2

n_data_clusters = 10

budget = 0.1

cluster_logger = ExperimentLogger()
classification_logger = ExperimentLogger()

training_window_size = 1000

X_training_window = collections.deque(maxlen=training_window_size)
y_training_window = collections.deque(maxlen=training_window_size)

stream = []

random_state = np.random.RandomState(random_number)

# ----------------------------------------------------------------------------------------------------------------------
# Initializations
# ----------------------------------------------------------------------------------------------------------------------

# Init data

# Iris Dataset

iris = sklearn.datasets.load_iris()
X_stream = iris.data[:, :2]
y_stream = iris.target

dataset = [(x, y_stream[i]) for i, x in enumerate(X_stream)]

X_stream = np.empty((stream_length + n_init_training, n_features))
y_stream = np.empty(stream_length + n_init_training, dtype=int)

for i in range(stream_length + n_init_training):
    X_tmp, y_tmp = next_sample(dataset)
    X_stream[i] = X_tmp
    y_stream[i] = int(y_tmp)

X_stream = np.apply_along_axis(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=0, arr=X_stream)
# Blob Dataset

# stream = sklearn.datasets.make_blobs(n_samples=stream_length + n_init_training, centers=n_data_clusters, n_features=2)[0]

# np.random.shuffle(dataset)

train_data = None

# Hyperplane Datasetgenerator

dataGenerator = HyperplaneGenerator(random_state=np.random.RandomState(random_number), n_features=2, mag_change=0)
dataGeneratorChange = HyperplaneGenerator(random_state=np.random.RandomState(random_number + 1), n_features=2, mag_change=0)

#X_stream, y_stream = dataGenerator.next_sample(stream_length + n_init_training)

X_stream, y_stream = dataGenerator.next_sample(int((stream_length + n_init_training)/2))
X_tmp, y_tmp = dataGeneratorChange.next_sample(int((stream_length + n_init_training) / 2))

X_stream = np.concatenate((X_stream, X_tmp), axis=0)
y_stream = np.concatenate((y_stream, y_tmp), axis=0)

#for i, x_t in enumerate(X_stream):
#    if X_stream[i][1] > 0.8:
#        y_stream[i] = 2

# init Clustering
cluStream = CluStream(n_micro_clusters=n_clusters, n_init_train=n_init_training, init_train=train_data)

# cluStream = CluStream(n_clusters, 1, n_init_training, train_data)


classes = np.unique(y_stream)

# '''query_strategy = StreamRandomSampling(budget=budget, random_state=get_randomseed(random_state))'''
# query_strategy = VariableUncertainty(budget=budget, random_state=get_randomseed(random_state))


# Init Classifier
classifiers = {
    'TraditionalBatch': ParzenWindowClassifier(classes=classes, random_state=get_randomseed(random_state)),
    'TraditionalIncremental': SklearnClassifier(GaussianNB(), classes=classes, random_state=get_randomseed(random_state)),
    'ClusteringBased': CluStreamClassifier(
        estimator_clf=SklearnClassifier(GaussianNB(), classes=classes, random_state=get_randomseed(random_state)),
        clustering=cluStream)
}

accuracy = []

for clf_name, classifier in classifiers.items():
    cluStream.reset()
    if clf_name.endswith('Incremental'):
        metric = pairwise_kernels
    else:
        metric = None

    fit_clf = False

    # init Active Learning strategy

    classes = np.unique(y_stream)

    query_strategy = StreamProbabilisticAL(random_state=get_randomseed(random_state),
                                           budget_manager=BalancedIncrementalQuantileFilter(),
                                           metric=metric,
                                           ##budget=budget
                                           )

    # Pretrain classifier
    pretrainX = X_stream[:n_init_training]
    pretrainY = y_stream[:n_init_training]

    classifier.fit(pretrainX, pretrainY)

    (X, y) = X_stream[1], y_stream[1]
    (X, y) = next_sample(dataset)

    X_training_window.extend(pretrainX)
    y_training_window.extend(pretrainY)

    # Label and Cluster Stream
    for i in range(n_init_training, stream_length + n_init_training):
        # Iris
        # (X, y) = next_sample(dataset)

        # Hyperplane
        (X, y) = (X_stream[i], y_stream[i])

        stream.append((X, y))

        # if cluStream.initialized:
        #    clusterId, _ = cluStream.nearest_cluster(X)
        #    labeled_samples_c = cluStream.micro_clusters[clusterId].labeled_samples
        #    X_cluster = [sample for sample in labeled_samples_c[0]]
        #    y_cluster = labeled_samples_c[1]
        #    #X_cluster, y_cluster = labeled_samples_c[:, 0].reshape(-1, 1), labeled_samples_c[:, 1]

        al_label = label_sample(X_cand=X,
                                y_cand=y,
                                query_strategy=query_strategy,
                                classifier=classifier,
                                X=X_training_window,
                                y=y_training_window,
                                # X=np.array([X]),
                                # y=np.array([[y]])
                                )
        #cluster_id = cluStream.fit_one(X, al_label)

        if clf_name.endswith('Batch'):
            classifier.fit(X_training_window, y_training_window)
            cluStream.fit_one(X, al_label)
        elif clf_name.startswith('Clustering'):
            classifier.partial_fit(X.reshape([1, -1]), np.array([al_label]))
        else:
            cluStream.fit_one(X, al_label)
            if not np.isnan(al_label):
                classifier.partial_fit(X.reshape([1, -1]), np.array([al_label]))


        # Only labeled instances in window
        # if not np.isnan(al_label):
        # X_pred_window.append(X)
        # y_pred_window.append(y)

        # Missing Labels in window
        X_training_window.append(X)
        #y_training_window.append(al_label)
        y_training_window.append(al_label)

        prediction = classifier.predict([X])[0]

        classification_logger.track_timestep(i)
        classification_logger.track_y(prediction)
        classification_logger.track_gt(y)
        accuracy.append(prediction == y)
        #classification_logger.track_accuracy(sum(accuracy) / len(accuracy))
        classification_logger.track_classifier(clf_name)
        classification_logger.finalize_round()

    print("Classifier: ", clf_name, "Accuracy: ", np.average(accuracy))
    # df_classification[ACCURACY] = accuracy

    for i, cluster_samples in enumerate(cluStream.cluster_test):
        if np.shape(cluster_samples) == (2,):
            cluster_samples = cluster_samples.reshape((1, 2))
        for (X_tmp, y_tmp) in cluster_samples:
            cluster_logger.track_cluster(i)
            cluster_logger.track_x1(X_tmp[0])
            cluster_logger.track_x2(X_tmp[1])
            cluster_logger.track_y(y_tmp)
            cluster_logger.track_classifier(clf_name)
            if not np.isnan(y_tmp):
                cluster_logger.track_gt(y_tmp)
            else:
                cluster_logger.track_gt(find_label(stream, X_tmp))
            cluster_logger.finalize_round()

    clusters = [c for i, c in cluStream.micro_clusters.items()]

    radi = np.array([mc.radius() for mc in clusters], dtype=object)

    center = np.array([mc.center for mc in clusters], dtype=object)

    ns = [mc.features["n"] for i, mc in cluStream.micro_clusters.items()]
    n = sum(ns)

    stds = np.ones((n_clusters, 2))
    stds_m = np.ones((n_clusters, 2))
    time_stamps = np.ones((n_clusters, 2))
    for i, mc in cluStream.micro_clusters.items():
        test = mc.relevance_stamp()
        time_stamps[i] = (i, mc.relevance_stamp())

# Saving Data
df_clustering = cluster_logger.get_dataframe()
df_classification = classification_logger.get_dataframe()

accuracy = gaussian_filter1d(np.array(accuracy, dtype=float), 100)

df_classification[ACCURACY] = accuracy

fp_clustering = "target/clustering.csv"
fp_classification = "target/classification.csv"
df_clustering.to_csv(fp_clustering, index=False)
df_classification.to_csv(fp_classification, index=False)


def map_edgecolor(val):
    if np.isnan(val):
        return "white"
    else:
        return "red"


def map_linewidth(val):
    if np.isnan(val):
        return 0.5
    else:
        return 1


# Plotting
sb.set_theme()

x_tmp = [tuple(t[0]) for t in stream]
d = dict(collections.Counter(x_tmp))

# Plot Clustering
for clf_name, classifier in classifiers.items():
    # Plotting Clusters
    fig1, axis = plt.subplots(1, 1)
    axis.set_title(clf_name)

    df_plot = df_clustering.loc[df_clustering[CLASSIFIER] == clf_name]
    sb.scatterplot(data=df_plot.loc[np.isnan(df_plot[Y])], x=X1, y=X2, palette="deep", hue=GT, ax=axis,
                   edgecolor='White',
                   linewidth=0.5)
    sb.scatterplot(data=df_plot.loc[~np.isnan(df_plot[Y])], x=X1, y=X2, palette="deep", hue=GT, ax=axis,
                   edgecolor='red', linewidth=1,
                   zorder=10, legend=False)

    # Add Cluster Radi
    for i, c in enumerate(center):
        axis.add_patch(patches.Circle(c, radius=radi[i] * cluStream.r_factor, fill=False, color="turquoise"))

    # Plotting Class distributions in Cluster
    sb.displot(data=df_plot, x=CLUSTER, hue=Y, palette="tab10", multiple="stack", bins=n_clusters)

# fi2 = plt.subplots()
# Plotting Class distribution in space
# sb.scatterplot(data=df, x=X1, y=X2, palette="deep", hue=Y)

# Plot Classification accuracy

g = sb.relplot(
    data=df_classification, x=TIMESTEP, y=ACCURACY,
    kind="line",
    hue=CLASSIFIER
)

save_image('target/test.pdf')
# plt.savefig('target/test.pdf')
# plt.show()
