import multiprocessing

from skmultiflow.trees import HoeffdingTreeClassifier

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

# Datasets
ABALONE_BIN = 720
ABALONE = 44956
COVERTYPE = 1596
HYPERPLANE = -1


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


def generate_data(dataset_id, init_train_length, shuffle, random_state, n_features=0):
    if dataset_id == HYPERPLANE:
        assert n_features > 0, "Please specify the number of features for the hyperplane generator"
        dataGenerator = HyperplaneGenerator(random_state=random_state,
                                            n_features=2,
                                            mag_change=0)
    else:
        rng = np.random.default_rng(random_state)
        dataGenerator = OpenMlStreamGenerator(dataset_id, shuffle=shuffle, rng=rng)
    # Generating Datastream
    stream_length = len(dataGenerator.y) - init_train_length - 1
    X, y = dataGenerator.next_sample(stream_length + init_train_length)
    return X, y


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
    run_algorithms = False
    target_directory = 'target'
    csv_filepath = os.path.join(target_directory, 'output.csv')

    datasetId = ABALONE_BIN
    # number of instances that are provided to the classifier
    init_train_length = 10
    # the length of the data stream
    stream_length = 1000
    # the size of the sliding window that limits the training data
    training_size = 300
    # the parameter dedicated to decide if the classifier needs to be refited with X and y.
    fit_clf = False

    n_cluster = 100
    n_budget = 10
    init_budget = 0.01
    n_reps = 30
    n_bandwidths = 1
    bandwidth_step_size = 0.5
    init_bandwidth = 1
    n_approaches = 3

    logger = CluStreamPerformanceLogger

    res = [0] * n_bandwidths * n_budget * n_approaches * n_reps
    args = [0] * n_bandwidths * n_budget * n_approaches * n_reps
    # I was checking if your way of creating the indices causes problems (it seems to be fine)
    # Why not simply defining res = [] and args = [] and then appending to these lists?
    all_used_indices = []

    # It might be easier (and better readable) to create a parameter grid
    # (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html#sklearn.model_selection.ParameterGrid)
    # convert it to a list and then simply loop over the grid

    # Looping over number of repetition
    if run_algorithms:
        for rep in range(n_reps):

            budget = init_budget
            random_state = rep

            # Generating Datastream
            X, y = generate_data(datasetId, init_train_length, shuffle=True, random_state=random_state)

            # Looping over n_budget budgets with stepsize 0.1
            for k in range(n_budget):

                bandwidth = init_bandwidth

                for i in range(n_bandwidths):

                    # Bandiwdths for Kernels, passed to Query Strategy and/or Classifier
                    metric_dict = {
                        'gamma': bandwidth
                    }

                    classes = np.unique(y)

                    # Different Approaches, defined by a tuple (Query Strategy, CLassifier)
                    query_strategies = {
                        'TraditionalBatch': (StreamProbabilisticAL(random_state=random_state, budget=budget,
                                                                   metric_dict=metric_dict),
                                             # VariableUncertainty(random_state=random_state),
                                             ParzenWindowClassifier(classes=classes,
                                                                    random_state=random_state,
                                                                    metric_dict=metric_dict, missing_label=None)),
                        'TraditionalIncremental':
                            (StreamProbabilisticAL(random_state=random_state, metric="rbf",
                                                   budget=budget, metric_dict=metric_dict),
                             # VariableUncertainty(random_state=random_state),
                             SklearnClassifier(HoeffdingTreeClassifier(), classes=classes,
                                               random_state=random_state, missing_label=None)),
                        'ClusteringBased': (StreamProbabilisticAL(random_state=random_state, budget=budget),
                                            # VariableUncertainty(random_state=random_state),
                                            CluStreamClassifier(estimator_clf=SklearnClassifier(
                                                HoeffdingTreeClassifier(),
                                                missing_label=None,
                                                classes=classes,
                                                random_state=random_state),
                                                metric_dict=metric_dict,
                                                missing_label=None))
                    }
                    assert len(query_strategies) == n_approaches, "Number of approaches does not match n_approaches"

                    for l, (query_strategy_name, (query_strategy, clf)) in enumerate(query_strategies.items()):
                        index = rep * (n_budget * n_bandwidths * len(query_strategies)) + (
                                k * n_bandwidths * len(query_strategies)) + (i * len(query_strategies)) + l
                        assert index not in all_used_indices, "We have an index overlap!"
                        all_used_indices.append(index)
                        args[index] = [X, y,
                                       query_strategy_name, query_strategy,
                                       clf, logger,
                                       training_size, init_train_length,
                                       rep, bandwidth]

                        # Sequential execution for debuggin
                        # res[index] = run(X, y, query_strategy_name, query_strategy, clf, logger, training_size, init_train_length, j, bandwidth)

                    bandwidth += bandwidth_step_size
                    bandwidth = np.round(bandwidth, 2)
                budget += 0.02

        # Parallel execution of run()
        results = run_async(run, args, multiprocessing.cpu_count() - 1)
        df = pd.concat(results)
        os.makedirs(target_directory, exist_ok=True)
        df.to_csv(csv_filepath, index=False)
    else:
        df = pd.read_csv(csv_filepath)

    sb.set_theme()

    # g = sb.relplot(
    #    data=df, x=TIMESTEP, y=ACCURACY,
    #    kind="line", col=BANDWIDTH, col_wrap=3,
    #    hue=CLASSIFIER, errorbar=None
    # )

    mean_rolling_avg = df.groupby([BANDWIDTH, CLASSIFIER, REP, BUDGET])[ACCURACY].rolling(30).mean().reset_index()
    df_budget = pd.DataFrame(mean_rolling_avg)
    bd_plot = init_bandwidth

    # Add mean rolling average as horizontal line and text
    # for i, ax in enumerate(g.axes.flat):
    #    avgs = mean_rolling_avg[bd_plot]
    #    for j, ((classifier, rep), mean_accuracy) in enumerate(avgs.items()):
    #        ax.text(x=100, y=0.6 - j * (0.03), s=f"{classifier}: {mean_accuracy:.3f}", color='black', ha='left',
    #                va='center', fontweight='bold')
    #    bd_plot += bandwidth_step_size
    #    bd_plot = np.round(bd_plot, 2)

    # Plotting Accuracy in respect to budget
    f = sb.relplot(
        data=df_budget, x=BUDGET, y=ACCURACY,
        col=CLASSIFIER, col_wrap=3,
        kind="line", hue=CLASSIFIER
    )

    image_filepath = os.path.join(target_directory, 'output.pdf')

    save_image(image_filepath)
