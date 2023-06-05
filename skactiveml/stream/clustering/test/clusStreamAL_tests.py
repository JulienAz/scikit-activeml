import multiprocessing

from skmultiflow.trees import HoeffdingTreeClassifier

from skactiveml.stream.clustering import CluStream
from skactiveml.stream.clustering.data.datasets import ABALONE_BIN, COVERTYPE, generate_data, HYPERPLANE, IRIS, \
    ELECTRICITY, INTERCHANGING_RBF, CHESSBOARD
from skactiveml.stream.clustering.test.stream_runner import *
from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import CluStreamPerformanceLogger, \
    ACCURACY, BUDGET, CLASSIFIER, REP, BANDWIDTH
from skactiveml.stream.clustering.util import save_image, run_async

import numpy as np
import pandas as pd

import seaborn as sb

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.classifier._clu_stream_classifier import CluStreamClassifier
from skactiveml.stream import StreamProbabilisticAL, StreamRandomSampling

import os


if __name__ == '__main__':
    target_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'target')
    csv_filepath = os.path.join(target_directory, 'output.csv')

    dataset = CHESSBOARD

    # number of instances that are provided to the classifier
    init_train_length = 10
    # the length of the data stream
    stream_length = 5000

    # Configurable when using Hyperplane
    n_features = 2
    mag_change = 0.2

    # the size of the sliding window that limits the training data
    training_size = 300
    # the parameter dedicated to decide if the classifier needs to be refited with X and y.
    fit_clf = False

    # Influences when clusters are deleted due to irrelevance (High -> less deletions/ low -> many deletions)
    clu_time_windows = [500, 1000, 1500, np.inf]

    shuffle_data = True

    n_cluster = 10
    n_budget = 2
    init_budget = 0.01
    budget_step_size = 0.1
    n_reps = 2
    n_bandwidths = 1
    bandwidth_step_size = 0.5
    init_bandwidth = 1
    n_approaches = 1

    base_classifier = HoeffdingTreeClassifier

    logger = CluStreamPerformanceLogger

    res = [0] * n_bandwidths * n_budget * n_approaches * n_reps * len(clu_time_windows)
    args = []
    assert n_cluster <= init_train_length
    # It might be easier (and better readable) to create a parameter grid
    # (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html#sklearn.model_selection.ParameterGrid)
    # convert it to a list and then simply loop over the grid

    # Looping over number of repetition
    for rep in range(n_reps):

        budget = init_budget
        random_state = rep

        # Generating Datastream
        X, y = generate_data(dataset, init_train_length, shuffle=shuffle_data, random_state=random_state, stream_length=stream_length, n_features=n_features, mag_change=mag_change)

        # Looping over n_budget budgets with stepsize 0.1
        for k in range(n_budget):

            bandwidth = init_bandwidth

            for i in range(n_bandwidths):
                # Bandiwdths for Kernels, passed to Query Strategy and/or Classifier
                metric_dict = {
                    'gamma': bandwidth
                }
                for c, clu_time_window in enumerate(clu_time_windows):
                    classes = np.unique(y)

                    # Init Clustering
                    clustering = CluStream(
                        n_micro_clusters=n_cluster,
                        n_init_train=init_train_length,
                        time_window=clu_time_window
                    )

                    # Different Approaches, defined by a tuple (Query Strategy, CLassifier)
                    query_strategies = {
                        'ClusteringBased': (StreamProbabilisticAL(random_state=random_state, budget=budget),
                                            # VariableUncertainty(random_state=random_state),
                                            CluStreamClassifier(estimator_clf=SklearnClassifier(
                                                base_classifier(),
                                                missing_label=None,
                                                classes=classes,
                                                random_state=random_state),
                                                clustering=clustering,
                                                metric_dict=metric_dict,
                                                missing_label=None)),
                    }
                    assert len(query_strategies) == n_approaches, "Number of approaches does not match n_approaches"

                    for l, (query_strategy_name, (query_strategy, clf)) in enumerate(query_strategies.items()):
                        index = rep * (n_budget * n_bandwidths * len(query_strategies)) + (
                                k * n_bandwidths * len(query_strategies)) + (i * len(query_strategies)) + l
                        args.append([X, y,
                                       query_strategy_name, query_strategy,
                                       clf, logger, dataset['name'],
                                       training_size, init_train_length,
                                       rep, bandwidth])

                        # Sequential execution for debuggin
                        res[index] = run(X, y, query_strategy_name, query_strategy, clf, logger, dataset['name'], training_size, init_train_length, rep, bandwidth)
                bandwidth += bandwidth_step_size
                bandwidth = np.round(bandwidth, 2)
            budget = min(budget + budget_step_size, 1.0)
    args = np.array(args, dtype=object)
    # Parallel execution of run()
    results = run_async(run, args, multiprocessing.cpu_count() - 1)
    df = pd.concat(results)
    os.makedirs(target_directory, exist_ok=True)
    df.to_csv(csv_filepath, index=False)