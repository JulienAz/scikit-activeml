import cProfile
from datetime import time
from multiprocessing import Pool
from random import random

import random
from time import sleep

import numpy as np
import pandas as pd
import sklearn.datasets
import matplotlib.pyplot as plt
from collections import deque

import seaborn as sb

from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import pairwise_kernels
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KernelDensity
from skmultiflow.data import HyperplaneGenerator

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.classifier._clu_stream_classifier import CluStreamClassifier

from skactiveml.stream import StreamRandomSampling, PeriodicSampling
from skactiveml.stream import FixedUncertainty, VariableUncertainty, Split, StreamProbabilisticAL
from skactiveml.stream.clustering.test.stream_runner import run
from skactiveml.utils import call_func

# number of instances that are provided to the classifier
init_train_length = 10
# the length of the data stream
stream_length = 1000
# the size of the sliding window that limits the training data
training_size = 100
# the parameter dedicated to decide if the classifier needs to be refited with X and y.
fit_clf = False

n_features = 2

n_reps = 1

# random state that is used to generate random seeds
random_number = 20

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
    REP,
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

logger = ExperimentLogger()
accuracy = []

df = pd.DataFrame(all_ids)
for rep in range(n_reps):
    random_state = np.random.RandomState(random_number + rep)
    #X, y = sklearn.datasets.make_classification(n_samples=init_train_length + stream_length, random_state=get_randomseed(random_state), shuffle=True)

    dataGenerator = HyperplaneGenerator(random_state=get_randomseed(random_state), n_features=2, mag_change=0)
    X, y = dataGenerator.next_sample(stream_length + init_train_length)

    #dataGenerator = HyperplaneGenerator(random_state=get_randomseed(random_state), n_features=2, mag_change=0)
    #dataGeneratorChange = HyperplaneGenerator(
    #    random_state=get_randomseed(random_state=np.random.RandomState(random_number + rep + 10)), n_features=2, mag_change=0)

    #X, y = dataGenerator.next_sample(int(stream_length / 2) + init_train_length)
    #X_tmp, y_tmp = dataGeneratorChange.next_sample(int(stream_length / 2))

    #X = np.concatenate((X, X_tmp), axis=0)
    #y = np.concatenate((y, y_tmp), axis=0)

    # Iris Dataset
    '''
    iris = sklearn.datasets.load_iris()
    X_stream = iris.data[:, :2]
    y_stream = iris.target

    dataset = [(x, y_stream[i]) for i, x in enumerate(X_stream)]

    X = np.empty((stream_length + init_train_length, n_features))
    y = np.empty(stream_length + init_train_length, dtype=int)

    for i in range(stream_length + init_train_length):
       X_tmp, y_tmp = next_sample(dataset)
       X[i] = X_tmp
       y[i] = int(y_tmp)

    X = np.apply_along_axis(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=0, arr=X)
    '''

    X_init = X[:init_train_length, :]
    y_init = y[:init_train_length]
    X_stream = X[init_train_length:, :]
    y_stream = y[init_train_length:]

    classes = np.unique(y_stream)

    clf_factory = lambda: ParzenWindowClassifier(classes=classes, random_state=get_randomseed(random_state))
    query_strategies = {
        #'StreamRandomSampling': StreamRandomSampling(random_state=get_randomseed(random_state)),
        #'PeriodicSampling': PeriodicSampling(random_state=get_randomseed(random_state)),
        #'FixedUncertainty': FixedUncertainty(random_state=get_randomseed(random_state)),
        #'VariableUncertainty': VariableUncertainty(random_state=get_randomseed(random_state)),
        #'Split': Split(random_state=get_randomseed(random_state)),
        'TraditionalBatch': (StreamProbabilisticAL(random_state=get_randomseed(random_state)),
                                  clf_factory()),
        'TraditionalIncremental': (StreamProbabilisticAL(random_state=get_randomseed(random_state), metric=pairwise_kernels),
                            SklearnClassifier(GaussianNB(), classes=classes, random_state=get_randomseed(random_state))),
        'ClusteringBased': (StreamProbabilisticAL(random_state=get_randomseed(random_state)),
                            CluStreamClassifier(estimator_clf=SklearnClassifier(GaussianNB(), classes=classes, random_state=get_randomseed(random_state))))
    }

    #results = [run(X_stream, y_stream, query_strategy_name, query_strategy, clf, ExperimentLogger, training_size, init_train_length) for query_strategy_name, (query_strategy, clf) in query_strategies.items()]
    #df = pd.concat(df, pd.concat(results))

    for query_strategy_name, (query_strategy, clf) in query_strategies.items():
        tmp_accuracy = []
        # initializing the training data
        X_train = deque(maxlen=training_size)
        X_train.extend(X_init)
        y_train = deque(maxlen=training_size)
        y_train.extend(y_init)
        # train the model with the initially available data
        clf.fit(X_train, y_train)
        # initialize the list that stores the result of the classifier's prediction
        correct_classifications = []
        count = 0
        for t, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):
            # create stream samples
            X_cand = x_t.reshape([1, -1])
            y_cand = y_t

            correct_classifications.append(clf.predict(X_cand)[0] == y_cand)
            # check whether to sample the instance or not
            # call_func is used since a classifier is not needed for RandomSampling and PeriodicSampling
            kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X_train)
            X_candidate_density = np.array([kde.score(X_cand)])

            sampled_indices, utilities = call_func(query_strategy.query, candidates=X_cand, X=X_train, y=y_train,
                                                   #utility_weight=X_candidate_density,
                                                   clf=clf, return_utilities=True, fit_clf=fit_clf)
            # create budget_manager_param_dict for BalancedIncrementalQuantileFilter used by StreamProbabilisticAL
            budget_manager_param_dict = {"utilities": utilities}
            # update the query strategy and budget_manager to calculate the right budget
            call_func(query_strategy.update, candidates=X_cand, queried_indices=sampled_indices, budget_manager_param_dict=budget_manager_param_dict)
            # count the number of queries
            count += len(sampled_indices)
            # add X_cand to X_train
            X_train.append(x_t)
            # add label or missing_label to y_train
            al_label = y_cand if len(sampled_indices) > 0 else clf.missing_label
            y_train.append(y_cand if len(sampled_indices) > 0 else clf.missing_label)

            if query_strategy_name.endswith('Batch'):
                clf.fit(X_train, y_train)
            elif query_strategy_name.startswith('Clustering'):
                clf.partial_fit(X_cand.reshape([1, -1]), np.array([al_label]))
            else:
                if not np.isnan(al_label):
                    clf.partial_fit(X_cand.reshape([1, -1]), np.array([al_label]))



            prediction = clf.predict(X_cand)[0]

            logger.track_timestep(t)
            logger.track_rep(rep)
            logger.track_y(prediction)
            logger.track_gt(y_cand)
            tmp_accuracy.append(prediction == y_cand)
            # classification_logger.track_accuracy(sum(accuracy) / len(accuracy))
            logger.track_classifier(query_strategy_name)
            logger.finalize_round()

        tmp_accuracy = gaussian_filter1d(np.array(tmp_accuracy, dtype=float), 100)
        accuracy = np.concatenate((accuracy, tmp_accuracy))
        # calculate and show the average accuracy
        print("Repitition: ", rep, "Query Strategy: ", query_strategy_name, ", Avg Accuracy: ", np.mean(correct_classifications), ", Acquisition count:", count)
        #plt.plot(gaussian_filter1d(np.array(correct_classifications, dtype=float), 50), label=query_strategy_name)

df = logger.get_dataframe()

#accuracy = gaussian_filter1d(np.array(accuracy, dtype=float), 100)

df[ACCURACY] = accuracy

fp_performance = "target/test_performance_abrupt.csv"
df.to_csv(fp_performance, index=False)

sb.set_theme()

g = sb.relplot(
    data=df, x=TIMESTEP, y=ACCURACY,
    kind="line",
    hue=CLASSIFIER
)

save_image('target/performance_test_abrupt.pdf')
#plt.legend()
#plt.show()