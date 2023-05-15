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

bandwidth = 2.2

# random state that is used to generate random seeds
random_number = 15

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

    random_state = np.random.RandomState(random_number)

    logger = CluStreamPerformanceLogger

    # Hyperplane Generator

    res = [0] * n_reps

    for j in range(n_reps):
        dataGenerator = HyperplaneGenerator(random_state=get_randomseed(random_state), n_features=2, mag_change=0)

        random_state = np.random.RandomState(random_number + j)
        # Abalone binary 50/50
        dataSetId = 720

        # Abalone
        # datasetId = 44956

        # Covertype
        # dataSetId = 1596

        dataGenerator = OpenMlStreamGenerator(dataSetId)
        stream_length = len(dataGenerator.y) - init_train_length - 1

        X, y = dataGenerator.next_sample(stream_length + init_train_length)

        args = [0] * n_budget
        budget = 0.1
        metric_dict = {
            'gamma': bandwidth
        }

        for i in range(n_budget):
            classes = np.unique(y)

            clf_factory = lambda: ParzenWindowClassifier(classes=classes, random_state=get_randomseed(random_state),
                                                         metric_dict=metric_dict)
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
                     SklearnClassifier(GaussianNB(), classes=classes, random_state=get_randomseed(random_state))),
                'ClusteringBased': (StreamProbabilisticAL(random_state=get_randomseed(random_state), budget=budget),
                                    CluStreamClassifier(estimator_clf=SklearnClassifier(GaussianNB(), classes=classes,
                                                                                        random_state=get_randomseed(
                                                                                            random_state)),
                                                        metric_dict=metric_dict))
            }
            args[i] = [query_strategies, X, y, logger, training_size, init_train_length, j, bandwidth]
            budget += 0.1
            budget = np.round(budget, 1)

        results = run_async(run_multiple, args, n_reps)
        res[j] = pd.concat(results)
    df = pd.concat(res)

    # test_result = run_multiple(query_strategies, X, y, logger, 0, training_size, init_train_length)

    # df[ACCURACY] = accuracy

    fp_performance = "target/performance_test_abalone.csv"  #MH: I would suggest using Parquet files since they are much smaller and also load faster.
    df.to_csv(fp_performance, index=False)

    sb.set_theme()

    g = sb.relplot(
        data=df, x=TIMESTEP, y=ACCURACY,
        kind="line", col=BUDGET, col_wrap=3,
        hue=CLASSIFIER, errorbar=None
    )

    mean_rolling_avg = df.groupby([BUDGET, CLASSIFIER, REP])[ACCURACY].mean()  #MH: I am not sure how many entries the grouped data frame will have. Why doesn't it have one row if you group by all those three columns? Do you maybe want to average over BUDGET and CLASSIFIER only?
    df_budget = pd.DataFrame(mean_rolling_avg.reset_index())
    budget_plot = 0.1

    # Add mean rolling average as horizontal line and text
    for i, ax in enumerate(g.axes.flat):
        avgs = mean_rolling_avg[np.round(budget_plot, 1)]  #MH: What does this do? I don't think I understand. doesn't this say mean_rolling_average[0.1]?
        for j, ((classifier, rep), mean_accuracy) in enumerate(avgs.items()):
            ax.text(x=100, y=0.6 - j * (0.03), s=f"{classifier}: {mean_accuracy:.3f}", color='black', ha='left',
                    va='center', fontweight='bold')
        budget_plot += 0.1

    f = sb.relplot(
        data=df_budget, x=BUDGET, y=ACCURACY,
        kind="line", hue=CLASSIFIER, errorbar=None
    )
    save_image('target/performance_test_abalone.pdf')  #MH: Detail, but best practice would be to use os.path.join('target', 'performance_test_abalone.pdf') to avoid problems when switching between e.g. windows and mac.
