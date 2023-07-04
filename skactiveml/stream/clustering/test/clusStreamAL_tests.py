import multiprocessing

from skmultiflow.trees import HoeffdingTreeClassifier

from skactiveml.stream.clustering import CluStream
from skactiveml.stream.clustering._clu_stream_al import MicroClfCluster
from skactiveml.stream.clustering.data.datasets import ABALONE_BIN, COVERTYPE, generate_data, HYPERPLANE, IRIS, \
    ELECTRICITY, INTERCHANGING_RBF, CHESSBOARD, RBF_GENERATOR
from skactiveml.stream.clustering.test.stream_runner import *
from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import CluStreamPerformanceLogger, \
    ACCURACY, BUDGET, CLASSIFIER, REP, BANDWIDTH, CluStreamClusteringLogger
from skactiveml.stream.clustering.util import save_image, run_async

import numpy as np
import pandas as pd

import seaborn as sb

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.classifier._clu_stream_classifier import CluStreamClassifier, CluStreamEnsembleClassifier
from skactiveml.stream import StreamProbabilisticAL, StreamRandomSampling

import os


if __name__ == '__main__':
    target_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'target')
    csv_acc_filepath = os.path.join(target_directory, 'accuracy_time_window.csv')
    csv_clu_filepath = os.path.join(target_directory, 'clustering_time_window.csv')
    csv_clu_stat_filepath = os.path.join(target_directory, 'cluster_statistics_time_window.csv')

    dataset = ELECTRICITY

    # number of instances that are provided to the classifier
    init_train_length = 20
    # the length of the data stream
    stream_length = 10000
    stream_start_point = 0

    random_state = 14

    # Configurable when using Hyperplane
    n_features = 2
    mag_change = 0.2

    # the size of the sliding window that limits the training data
    training_size = 400
    # the parameter dedicated to decide if the classifier needs to be refited with X and y.
    fit_clf = False

    # Influences when clusters are deleted due to irrelevance (High -> less deletions/ low -> many deletions)
    clu_time_windows = [np.inf]

    # Number of clusters to be executed
    cluster_numbers = [3, 10, 20, 50]

    shuffle_data = False
    log_clustering = False
    log_clu_statistics = False

    n_budget = 10
    init_budget = 0.1
    budget_step_size = 0.1
    n_reps = 3
    n_bandwidths = 1
    bandwidth_step_size = 0.5
    init_bandwidth = 1

    base_classifier = HoeffdingTreeClassifier

    args = []
    res = []
    # It might be easier (and better readable) to create a parameter grid
    # (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html#sklearn.model_selection.ParameterGrid)
    # convert it to a list and then simply loop over the grid

    # Looping over number of repetition
    for rep in range(n_reps):

        budget = init_budget
        random_state += rep

        # Generating Datastream
        X, y = generate_data(dataset, init_train_length,
                             shuffle=shuffle_data, random_state=random_state, stream_length=stream_length,
                             start_point=stream_start_point, n_features=n_features, mag_change=mag_change)
        classes = np.unique(y)
        # Looping over n_budget budgets with stepsize 0.1
        for k in range(n_budget):

            bandwidth = init_bandwidth

            for i in range(n_bandwidths):
                # Bandiwdths for Kernels, passed to Query Strategy and/or Classifier
                metric_dict = {
                    'gamma': bandwidth
                }
                for c, clu_time_window in enumerate(clu_time_windows):
                    # Looping over Number of clusters configured
                    for n_cluster in cluster_numbers:
                        if n_cluster > init_train_length:
                            init_train_length = n_cluster

                        # Init Clustering params
                        clusteringParams = {
                            'n_micro_clusters': n_cluster,
                            'n_init_train': init_train_length,
                            'time_window': clu_time_window,
                            'classes': classes
                        }
                        clusteringEnsembleClfParams = {
                            'micro_cluster': MicroClfCluster,
                            'n_micro_clusters': n_cluster,
                            'n_init_train': init_train_length,
                            'time_window': clu_time_window,
                            'classes': classes
                        }

                        # Different Approaches, defined by a tuple (Query Strategy, CLassifier)
                        query_strategies = {
                            #'ClusteringIncremental': (StreamProbabilisticAL(random_state=random_state, budget=budget),
                            #                         # VariableUncertainty(random_state=random_state),
                            #                          CluStreamClassifier(estimator_clf=SklearnClassifier(
                            #                              base_classifier(),
                            #                              missing_label=None,
                            #                             classes=classes,
                            #                              random_state=random_state),
                            #                              clustering=clustering,
                            #                              metric_dict=metric_dict,
                            #                              missing_label=None)),
                            #'ClusteringClfReset': (StreamProbabilisticAL(random_state=random_state, budget=budget),
                            #                    # VariableUncertainty(random_state=random_state),
                            #                    CluStreamClassifier(estimator_clf=SklearnClassifier(
                            #                        base_classifier(),
                            #                        missing_label=None,
                            #                        classes=classes,
                            #                        random_state=random_state),
                            #                        clustering=clustering,
                            #                        metric_dict=metric_dict,
                            #                        missing_label=None,
                            #                        refit=True)),
                            'ClusteringClfRefit': (StreamProbabilisticAL(random_state=random_state, budget=budget),
                                                   # VariableUncertainty(random_state=random_state),
                                                   CluStreamClassifier(estimator_clf=SklearnClassifier(
                                                      base_classifier(),
                                                       missing_label=None,
                                                       classes=classes,
                                                       random_state=random_state),
                                                       metric_dict=metric_dict,
                                                       missing_label=None,
                                                       refit=True,
                                                       clustering_param_dict=clusteringParams
                                                   )),
                            'ClusteringClfEnsemble': (StreamProbabilisticAL(random_state=random_state, budget=budget),
                                                   # VariableUncertainty(random_state=random_state),
                                                   CluStreamEnsembleClassifier(estimator_clf=SklearnClassifier(
                                                       base_classifier(),
                                                       missing_label=None,
                                                       classes=classes,
                                                       random_state=random_state),
                                                       #clustering=clusteringEnsembleClf,
                                                       metric_dict=metric_dict,
                                                       missing_label=None,
                                                       refit=True,
                                                       clustering_param_dict=clusteringEnsembleClfParams
                                                   )),
                            #'ClusteringBatch': (StreamProbabilisticAL(random_state=random_state, budget=budget),
                            #                    # VariableUncertainty(random_state=random_state),
                            #                    CluStreamClassifier(estimator_clf=SklearnClassifier(
                            #                        base_classifier(),
                            #                        missing_label=None,
                            #                        classes=classes,
                            #                        random_state=random_state),
                            #                        clustering=clustering,
                            #                        metric_dict=metric_dict,
                           #                         missing_label=None)),
                        }

                        for l, (query_strategy_name, (query_strategy, clf)) in enumerate(query_strategies.items()):
                            index = rep * (n_budget * n_bandwidths * len(query_strategies)) + (
                                    k * n_bandwidths * len(query_strategies)) + (i * len(query_strategies)) + l
                            args.append([X, y,
                                         query_strategy_name, query_strategy,
                                         clf, dataset['name'],
                                         training_size, init_train_length,
                                         rep, bandwidth, log_clustering, log_clu_statistics])

                        # Sequential execution for debuggin
                        #res.append(run(X, y, query_strategy_name, query_strategy, clf, dataset['name'], training_size, init_train_length, rep, bandwidth, log_clustering, log_clu_statistics))
                bandwidth += bandwidth_step_size
                bandwidth = np.round(bandwidth, 2)
            budget = min(budget + budget_step_size, 1.0)
    args = np.array(args, dtype=object)
    # Parallel execution of run()
    results = run_async(run, args, multiprocessing.cpu_count() - 1)
    #results = res
    df_acc = pd.concat([t[0] for t in results])

    # Concatenate the second entries (df2)
    df_clu = pd.concat([t[1] for t in results])

    df_clu_statistics = pd.concat([t[2] for t in results])

    #df = pd.concat(results)

    os.makedirs(target_directory, exist_ok=True)

    df_acc.to_csv(csv_acc_filepath, index=False)
    if log_clustering:
        df_clu.to_csv(csv_clu_filepath, index=False)

    if log_clu_statistics:
        df_clu_statistics.to_csv(csv_clu_stat_filepath, index=False)