from collections import deque


import random

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import KernelDensity

from skactiveml.utils import call_func


def run(X, y, approach_name, query_strategy, clf, logger, n_training_size=100, n_init_traing=10, rep=0, band_width=0.1, fit_clf=False):
    logger = logger()

    # Dividing Pretraining and Stream data
    X_init = X[:n_init_traing, :]
    y_init = y[:n_init_traing]
    X_stream = X[n_init_traing:, :]
    y_stream = y[n_init_traing:]

    unique_classes = np.unique(y)

    tmp_accuracy = []

    # initializing the training window
    X_train = deque(maxlen=n_training_size)
    X_train.extend(X_init)
    y_train = deque(maxlen=n_training_size)
    y_train.extend(y_init)

    # pretrain
    clf.fit(X_train, y_train, classes=unique_classes)

    correct_classifications = []
    count = 0
    budget = query_strategy.budget

    for t, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):
        # create current sample
        X_cand = x_t.reshape([1, -1])
        y_cand = y_t

        # Classifier prediction
        prediction = clf.predict(X_cand)[0]

        correct_classifications.append(prediction == y_cand)

        # Kernel Densities (currently disabled)
        #kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X_train)  #MH: Don't you want to add the bandwidth parameter here?
        #X_candidate_density = np.array([kde.score(X_cand)])

        # Query decision
        sampled_indices, utilities = call_func(query_strategy.query, candidates=X_cand, X=X_train, y=y_train,
                                               # utility_weight=X_candidate_density,
                                               clf=clf, return_utilities=True, fit_clf=fit_clf)

        # create budget_manager_param_dict for BalancedIncrementalQuantileFilter
        budget_manager_param_dict = {"utilities": utilities}
        # update the query strategy and budget_manager to calculate the right budget
        call_func(query_strategy.update, candidates=X_cand, queried_indices=sampled_indices,
                  budget_manager_param_dict=budget_manager_param_dict)

        # count the number of queries
        count += len(sampled_indices)

        al_label = y_cand if len(sampled_indices) > 0 else clf.missing_label  #MH: In which cases is len(sampled_indices) > 1?

        # Append to Trainingwindow
        X_train.append(x_t)
        y_train.append(al_label)



        # Test with random sampling
        '''X_train.append(x_t)
        al_label = sample_random(y_cand, budget, clf.missing_label)
        y_train.append(al_label)
        count += 1 if al_label is not clf.missing_label else 0'''

        if approach_name.endswith('Batch'):
            clf.fit(X_train, y_train, classes=unique_classes)
        elif approach_name.startswith('Clustering'):
            # Missing Labels are handled in the Classifier itself
            clf.partial_fit(X_cand, np.array([al_label]))
        else:
            if not al_label is clf.missing_label:
                clf.partial_fit(X_cand, np.array([al_label]))

        logger.track_timestep(t)
        logger.track_y(prediction)
        logger.track_label(al_label)
        logger.track_gt(y_cand)
        logger.track_budget(budget)
        logger.track_bandwidth(band_width)
        logger.track_x1(X_cand[0][0])
        logger.track_x2(X_cand[0][1])
        logger.track_rep(rep)
        tmp_accuracy.append(prediction == y_cand)
        # classification_logger.track_accuracy(sum(accuracy) / len(accuracy))
        logger.track_classifier(approach_name)
        logger.finalize_round()

    df = logger.get_dataframe()
    acc_series = pd.Series(tmp_accuracy)
    accuracy = acc_series.rolling(window=30).mean()

    df["Accuracy"] = accuracy

    # calculate and show the average accuracy
    print("Repition", rep, "Query Strategy: ", approach_name, "Budget: ", budget, "Bandwidth: " , band_width, ", Avg Accuracy: ", np.mean(correct_classifications),
          ", Acquisition count:", count)

    return df


def run_multiple(query_strategies: dict, X, y, logger, n_training_size=0, n_init_traing=10, rep=0, bandwidth=0.1, fit_clf=False):

    results = [run(X, y, query_strategy_name, query_strategy, clf, logger, n_training_size, n_init_traing, rep, bandwidth)
               for query_strategy_name, (query_strategy, clf) in query_strategies.items()]
    return pd.concat(results)

# For Debugging
def run_sequential(X, y, approach_name, query_strategy, clf, logger, n_training_size=100, n_init_traing=10,  rep=0, band_width=0.1, fit_clf=False):
    logger = logger()

    X_init = X[:n_init_traing, :]
    y_init = y[:n_init_traing]
    X_stream = X[n_init_traing:, :]
    y_stream = y[n_init_traing:]
    y_stream_al = sample(X, y[n_init_traing:], clf.missing_label, query_strategy.budget)

    X_train = deque(maxlen=n_training_size)
    X_train.extend(X_init)
    y_train = deque(maxlen=n_training_size)
    y_train.extend(y_init)

    clf.fit(X_train, y_train)

    '''
    for i, y_cand in enumerate(y_stream):
        if approach_name.endswith('Batch'):
            clf.fit(X, y_cand)  # MH: Here you use y_train, in line 78 you use al_label. Why?
        elif approach_name.startswith('Clustering'):
            clf.partial_fit(X[i].reshape([1, -1]), np.array([y_cand]))  # MH: (1) Not sure, do you have to reshape again? (probably yes) (2) What happens when we fit / partial_fit on missing labels? couldn't we simply skip it in this case?
        else:
            if not y_cand is clf.missing_label:
                clf.partial_fit(X[i].reshape([1, -1]), np.array([y_cand]))
    '''
    correct_classifications = []
    count = 0
    budget = query_strategy.budget

    tmp_accuracy = []

    for t, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):
        # create stream samples
        X_cand = x_t.reshape([1, -1])
        y_cand = y_t
        al_label = y_stream_al[t]

        prediction = clf.predict(X_cand)[0]
        correct_classifications.append(prediction == y_cand)

        X_train.append(x_t)
        y_train.append(al_label)
        # count the number of queries
        count += 1 if al_label is not clf.missing_label else 0

        # add label or missing_label to y_train
        #al_label = y_cand

        if approach_name.endswith('Batch'):
            clf.fit(X_train, y_train)  # MH: Here you use y_train, in line 78 you use al_label. Why?
        elif approach_name.startswith('Clustering'):
            clf.partial_fit(x_t.reshape([1, -1]), np.array([al_label]))
            # MH: (1) Not sure, do you have to reshape again? (probably yes) (2) What happens when we fit / partial_fit on missing labels? couldn't we simply skip it in this case?
        else:
            if not al_label is clf.missing_label:
                clf.partial_fit(x_t.reshape([1, -1]), np.array([al_label]))

        logger.track_y(prediction)
        logger.track_timestep(t)
        logger.track_label(al_label)
        logger.track_gt(y[t])
        logger.track_budget(budget)
        logger.track_bandwidth(band_width)
        logger.track_x1(X_cand[0][0])
        logger.track_x2(X_cand[0][1])
        logger.track_rep(rep)
        tmp_accuracy.append(prediction == y_cand)

        logger.track_classifier(approach_name)
        logger.finalize_round()

    df = logger.get_dataframe()
    # accuracy = gaussian_filter1d(np.array(tmp_accuracy, dtype=float), 100)  # MH: here you could do something like pd.Series(temp_accuracy).rolling(windowsize).mean()
    df["Accuracy"] = tmp_accuracy

    # calculate and show the average accuracy
    print("Repition", rep, "Query Strategy: ", approach_name, "Budget: ", budget, "Bandwidth: ", band_width,
          ", Avg Accuracy: ", np.mean(correct_classifications),
          ", Acquisition count:", count)

    return df


def sample(X, y, missing_label=None, budget=0.1):
    num_elements = len(y)
    num_none_values = int((1 - budget) * num_elements)  # Calculate the number of elements to set to None

    y_sampled = np.array(y, dtype=object)
    indices = list(range(num_elements))  # Create a list of indices from 0 to num_elements-1
    random.shuffle(indices)  # Shuffle the list of indices

    for i in range(num_none_values):
        index = indices[i]  # Get the i-th shuffled index
        y_sampled[index] = missing_label  # Set the element at the shuffled index to None

    return y_sampled

def sample_random(y, budget, missing_label=None):
    choices = [y, missing_label]
    probabilities = [budget, 1 - budget]
    return random.choices(choices, probabilities)[0]