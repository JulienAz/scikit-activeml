

from skactiveml.stream.clustering.test.stream_runner import run, run_multiple
from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import CluStreamPerformanceLogger
from skactiveml.stream.datagenerator._stream_generator import OpenMlStreamGenerator

from multiprocessing import Pool
from random import random

import random
import openml

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
from skactiveml.utils import call_func

from skmultiflow.trees import HoeffdingTreeClassifier

# number of instances that are provided to the classifier
init_train_length = 100
# the length of the data stream
stream_length = 5000
# the size of the sliding window that limits the training data
training_size = 300
# the parameter dedicated to decide if the classifier needs to be refited with X and y.
fit_clf = False

n_cluster = 100

budget = 0.2

n_features = 2

n_reps = 5

# random state that is used to generate random seeds
random_number = 25

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


def run_async(function, args_list, njobs, sleep_time_s=0.1):
    pool = Pool(njobs)
    results = {i: pool.apply_async(function, args=args)
               for i, args in enumerate(args_list)}
    while not all(future.ready() for future in results.values()):
        sleep(sleep_time_s)
    results = [results[i].get() for i in range(len(results))]
    pool.close()
    return results


if __name__ == '__main__':

    #test = openml.datasets.get_dataset(1596)
    #DataGenerator = OpenMlStreamGenerator(test)


    logger = CluStreamPerformanceLogger

    random_state = np.random.RandomState(random_number)

    dataGenerator = HyperplaneGenerator(random_state=get_randomseed(random_state), n_features=2, mag_change=0)

    # Abalone binary 50/50
    dataSetId = 720

    # Covertype
    #dataset = openml.datasets.get_dataset(1596)

    # Abalone
    #dataset = openml.datasets.get_dataset(44956)

    #Abalone 2 classes
    #dataset = openml.datasets.get_dataset(41357)


    dataGenerator = OpenMlStreamGenerator(dataSetId)
    stream_length = len(dataGenerator.y) - init_train_length - 1

    X, y = dataGenerator.next_sample(stream_length + init_train_length)

    #X, y_centers = sklearn.datasets.make_blobs(
    #    n_samples=init_train_length + stream_length,
    #    centers=30,
    #    random_state=get_randomseed(random_state),
    #    shuffle=True)
    #y = y_centers % 2

    classes = np.unique(y)

    clf_factory = lambda: ParzenWindowClassifier(classes=classes, random_state=get_randomseed(random_state))
    query_strategies = {
        # 'StreamRandomSampling': StreamRandomSampling(random_state=get_randomseed(random_state)),
        # 'PeriodicSampling': PeriodicSampling(random_state=get_randomseed(random_state)),
        # 'FixedUncertainty': FixedUncertainty(random_state=get_randomseed(random_state)),
        # 'VariableUncertainty': VariableUncertainty(random_state=get_randomseed(random_state)),
        # 'Split': Split(random_state=get_randomseed(random_state)),
        'TraditionalBatch': (StreamProbabilisticAL(random_state=get_randomseed(random_state),  budget=budget),
                             clf_factory()),
        'TraditionalIncremental':
                            (StreamProbabilisticAL(random_state=get_randomseed(random_state), metric=pairwise_kernels, budget=budget),
                            SklearnClassifier(HoeffdingTreeClassifier(), classes=classes, random_state=get_randomseed(random_state))),
        'ClusteringBased': (StreamProbabilisticAL(random_state=get_randomseed(random_state), budget=budget),
                            CluStreamClassifier(estimator_clf=SklearnClassifier(HoeffdingTreeClassifier(), classes=classes,

                                                                                random_state=get_randomseed(
                                                                                    random_state))))
    }
    args = [[X, y, approach_name, query_strategy, clf, logger, training_size, init_train_length]
    for approach_name, (query_strategy, clf) in query_strategies.items()]

    results = run_async(run, args, len(query_strategies))
    #results = [run(X, y, approach_name, query_strategy, clf, logger, training_size, init_train_length)  #
    #           for approach_name, (query_strategy, clf) in query_strategies.items()]
    df = pd.concat(results)

    #test_result = run_multiple(query_strategies, X, y, logger, 0, training_size, init_train_length)

    #df[ACCURACY] = accuracy

    fp_performance = "target/test_performance_abrupt.csv"
    #df.to_csv(fp_performance, index=False)

    sb.set_theme()

    g = sb.relplot(
        data=df, x=TIMESTEP, y=ACCURACY,
        kind="line",
        hue=CLASSIFIER
    )

    g.ax.text(.8, .85, np.mean(df[ACCURACY]), transform=g.ax.transAxes)

    save_image('target/performance_test_abrupt.pdf')