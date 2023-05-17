from skactiveml.stream.clustering.test.stream_runner import *
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

import os


# number of instances that are provided to the classifier
init_train_length = 10
# the length of the data stream
stream_length = 10000
# the size of the sliding window that limits the training data
training_size = 300
# the parameter dedicated to decide if the classifier needs to be refited with X and y.
fit_clf = False

n_cluster = 100

budget = 0.1

n_features = 2

n_budget = 9

n_reps = 1

n_bandwidths = 3

bandwidth_step_size = 0.1
init_bandwidth = 1

n_approaches = 3

# random state that is used to generate random seeds
random_number = 23

REP = "rep"
TIMESTEP = "round"
APPROACH = "Approach"
ACCURACY = "Accuracy"
CLUSTER = "Cluster"
CLASSIFIER = "Classifier"
TIME = "time"
BUDGET = "Budget"
BANDWIDTH = "Kernel_Bandwidth"
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

    random_state = np.random.RandomState(random_number)

    logger = CluStreamPerformanceLogger

    # Hyperplane Generator

    res = [0] * n_reps

    for j in range(n_reps):
        #random_state = np.random.RandomState(random_number + j)

        #dataGenerator = HyperplaneGenerator(random_state=get_randomseed(random_state), n_features=2, mag_change=0.2)


        # Abalone binary 50/50
        #dataSetId = 720

        # Abalone
        # datasetId = 44956

        # Covertype
        # dataSetId = 1596

        # dataGenerator = OpenMlStreamGenerator(dataSetId)
        # stream_length = len(dataGenerator.y) - init_train_length - 1

        #X, y = dataGenerator.next_sample(stream_length + init_train_length)

        args = [0] * n_bandwidths * n_budget * n_approaches
        bandwidth = init_bandwidth

        for k in range(n_budget):
            bandwidth = init_bandwidth
            for i in range(n_bandwidths):
                random_state = np.random.RandomState(random_number + j)

                dataGenerator = HyperplaneGenerator(random_state=get_randomseed(random_state), n_features=2,
                                                    mag_change=0.2)

                # Abalone binary 50/50
                # dataSetId = 720

                # Abalone
                # datasetId = 44956

                # Covertype
                # dataSetId = 1596

                # dataGenerator = OpenMlStreamGenerator(dataSetId)
                # stream_length = len(dataGenerator.y) - init_train_length - 1

                X, y = dataGenerator.next_sample(stream_length + init_train_length)

                metric_dict = {
                    'gamma': bandwidth
                }

                classes = np.unique(y)

                clf_factory = lambda: ParzenWindowClassifier(classes=classes, random_state=get_randomseed(random_state),
                                                             metric_dict=metric_dict, missing_label=None)
                query_strategies = {
                    # 'StreamRandomSampling': StreamRandomSampling(random_state=get_randomseed(random_state)),
                    # 'PeriodicSampling': PeriodicSampling(random_state=get_randomseed(random_state)),
                    # 'FixedUncertainty': FixedUncertainty(random_state=get_randomseed(random_state)),
                    # 'VariableUncertainty': VariableUncertainty(random_state=get_randomseed(random_state)),
                    # 'Split': Split(random_state=get_randomseed(random_state)),
                    'TraditionalBatch': (StreamProbabilisticAL(random_state=get_randomseed(random_state), budget=budget,
                                                               metric_dict=metric_dict),
                                         clf_factory()),
                    'TraditionalIncremental':
                        (StreamProbabilisticAL(random_state=get_randomseed(random_state), metric="rbf",
                                               budget=budget, metric_dict=metric_dict),
                         SklearnClassifier(GaussianNB(), classes=classes, random_state=get_randomseed(random_state), missing_label=None)),
                    'ClusteringBased': (StreamProbabilisticAL(random_state=get_randomseed(random_state), budget=budget),
                                        CluStreamClassifier(estimator_clf=SklearnClassifier(GaussianNB(), missing_label=None,
                                                                                            classes=classes,
                                                                                            random_state=get_randomseed(
                                                                                                random_state)),
                                                            metric_dict=metric_dict, missing_label=None))
                }
                for l, (query_strategy_name, (query_strategy, clf)) in enumerate(query_strategies.items()):
                    #args for sequential (First sample then train)
                    args[(k * n_bandwidths * len(query_strategies)) + (i * len(query_strategies)) + l] = [X, y, query_strategy_name, query_strategy, clf, logger, j, bandwidth]
                    #results = run_sequential(X, y, query_strategy_name, query_strategy, clf, logger, j, bandwidth)

                    #Common approach
                    args[(k * n_bandwidths * len(query_strategies)) + (i * len(query_strategies)) + l] = [X, y, query_strategy_name, query_strategy, clf, logger, training_size, init_train_length, j, bandwidth]
                bandwidth += bandwidth_step_size
                bandwidth = np.round(bandwidth, 2)
            budget += 0.1
            budget = np.round(budget, 1)
        results = run_async(run, args, n_bandwidths * n_budget * n_approaches)

    df = pd.concat(results)


    # df[ACCURACY] = accuracy

    target_directory = 'target'
    os.makedirs(target_directory, exist_ok=True)

    csv_filepath = os.path.join(target_directory, 'output.csv')

    df.to_csv(csv_filepath, index=False)

    sb.set_theme()

    #g = sb.relplot(
    #    data=df, x=TIMESTEP, y=ACCURACY,
    #    kind="line", col=BANDWIDTH, col_wrap=3,
    #    hue=CLASSIFIER, errorbar=None
    #)

    mean_rolling_avg = df.groupby([BANDWIDTH, CLASSIFIER, REP, BUDGET])[ACCURACY].mean()
    df_budget = pd.DataFrame(mean_rolling_avg.reset_index())
    bd_plot = init_bandwidth

    # Add mean rolling average as horizontal line and text
    #for i, ax in enumerate(g.axes.flat):
    #    avgs = mean_rolling_avg[bd_plot]
    #    for j, ((classifier, rep), mean_accuracy) in enumerate(avgs.items()):
    #        ax.text(x=100, y=0.6 - j * (0.03), s=f"{classifier}: {mean_accuracy:.3f}", color='black', ha='left',
    #                va='center', fontweight='bold')
    #    bd_plot += bandwidth_step_size
    #    bd_plot = np.round(bd_plot, 2)

    f = sb.relplot(
        data=df_budget, x=BUDGET, y=ACCURACY,
        col=BANDWIDTH, col_wrap=3,
        kind="line", hue=CLASSIFIER, errorbar=None
    )

    image_filepath = os.path.join(target_directory, 'output.pdf')

    save_image(image_filepath)
