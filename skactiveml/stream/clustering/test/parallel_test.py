import multiprocessing

from skmultiflow.trees import HoeffdingTreeClassifier

from skactiveml.stream.clustering.datasets import ABALONE_BIN,COVERTYPE, generate_data
from skactiveml.stream.clustering.test.stream_runner import *
from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import CluStreamPerformanceLogger, \
    ACCURACY, BUDGET, CLASSIFIER, REP, BANDWIDTH
from skactiveml.stream.clustering.util import save_image, run_async

import numpy as np
import pandas as pd

import seaborn as sb

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.classifier._clu_stream_classifier import CluStreamClassifier
from skactiveml.stream import StreamProbabilisticAL

import os


if __name__ == '__main__':
    target_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'target')
    csv_filepath = os.path.join(target_directory, 'output.csv')

    datasetId = COVERTYPE
    # number of instances that are provided to the classifier
    init_train_length = 10
    # the length of the data stream
    stream_length = 5000
    # the size of the sliding window that limits the training data
    training_size = 300
    # the parameter dedicated to decide if the classifier needs to be refited with X and y.
    fit_clf = False

    n_cluster = 100
    n_budget = 2
    init_budget = 0.01
    budget_step_size = 0.1
    n_reps = 1
    n_bandwidths = 1
    bandwidth_step_size = 0.5
    init_bandwidth = 1
    n_approaches = 3

    base_classifier = HoeffdingTreeClassifier

    logger = CluStreamPerformanceLogger

    res = [0] * n_bandwidths * n_budget * n_approaches * n_reps
    args = [0] * n_bandwidths * n_budget * n_approaches * n_reps

    # It might be easier (and better readable) to create a parameter grid
    # (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html#sklearn.model_selection.ParameterGrid)
    # convert it to a list and then simply loop over the grid

    # Looping over number of repetition
    for rep in range(n_reps):

        budget = init_budget
        random_state = rep

        # Generating Datastream
        X, y = generate_data(datasetId, init_train_length, shuffle=True, random_state=random_state, stream_length=stream_length)

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
                    #'TraditionalBatch': (StreamProbabilisticAL(random_state=random_state, budget=budget,
                    #                                           metric_dict=metric_dict),
                    #                     # VariableUncertainty(random_state=random_state),
                    #                     ParzenWindowClassifier(classes=classes,
                    #                                            random_state=random_state,
                    #                                            metric_dict=metric_dict, missing_label=None)),
                    'OPALBatch':
                        (StreamProbabilisticAL(random_state=random_state, metric="rbf",
                                               budget=budget, metric_dict=metric_dict),
                                         # VariableUncertainty(random_state=random_state),
                        SklearnClassifier(base_classifier(), classes=classes,
                                            random_state=random_state, missing_label=None)),
                    'OPALIncremental':
                        (StreamProbabilisticAL(random_state=random_state, metric="rbf",
                                               budget=budget, metric_dict=metric_dict),
                         # VariableUncertainty(random_state=random_state),
                         SklearnClassifier(base_classifier(), classes=classes,
                                           random_state=random_state, missing_label=None)),
                    'ClusteringBased': (StreamProbabilisticAL(random_state=random_state, budget=budget),
                                        # VariableUncertainty(random_state=random_state),
                                        CluStreamClassifier(estimator_clf=SklearnClassifier(
                                            base_classifier(),
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
                    args[index] = [X, y,
                                   query_strategy_name, query_strategy,
                                   clf, logger,
                                   training_size, init_train_length,
                                   rep, bandwidth]

                    # Sequential execution for debuggin
                    # res[index] = run(X, y, query_strategy_name, query_strategy, clf, logger, training_size, init_train_length, rep, bandwidth)

                bandwidth += bandwidth_step_size
                bandwidth = np.round(bandwidth, 2)
            budget += budget_step_size

    # Parallel execution of run()
    results = run_async(run, args, multiprocessing.cpu_count() - 1)
    df = pd.concat(results)
    os.makedirs(target_directory, exist_ok=True)
    df.to_csv(csv_filepath, index=False)