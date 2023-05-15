from collections import deque

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import KernelDensity

from skactiveml.utils import call_func


def run(X, y, approach_name, query_strategy, clf, logger, n_training_size=100, n_init_traing=10, rep=0, band_width=0.1, fit_clf=False):
    logger = logger()
    X_init = X[:n_init_traing, :]
    y_init = y[:n_init_traing]
    X_stream = X[n_init_traing:, :]
    y_stream = y[n_init_traing:]

    unique_classes = np.unique(y)

    '''
    for target_class in unique_classes:
        sample_extracted = False
        i = 0
        while not sample_extracted:
            X_sample, y_sample = X[i], y[i]
            if y_sample == target_class:
                X_init = np.concatenate((X_init, [X_sample]))
                y_init = np.concatenate((y_init, [y_sample]))
                sample_extracted = True
            i += 1
    '''

    tmp_accuracy = []
    # initializing the training data
    X_train = deque(maxlen=n_training_size)
    X_train.extend(X_init)
    y_train = deque(maxlen=n_training_size)
    y_train.extend(y_init)
    # train the model with the initially available data
    clf.fit(X_train, y_train)
    # initialize the list that stores the result of the classifier's prediction
    correct_classifications = []
    count = 0
    budget = query_strategy.budget

    for t, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):
        # create stream samples
        X_cand = x_t.reshape([1, -1])
        y_cand = y_t

        correct_classifications.append(clf.predict(X_cand)[0] == y_cand)
        # check whether to sample the instance or not
        # call_func is used since a classifier is not needed for RandomSampling and PeriodicSampling
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X_train)  #MH: Don't you want to add the bandwidth parameter here?
        X_candidate_density = np.array([kde.score(X_cand)])

        sampled_indices, utilities = call_func(query_strategy.query, candidates=X_cand, X=X_train, y=y_train,
                                               # utility_weight=X_candidate_density,
                                               clf=clf, return_utilities=True, fit_clf=fit_clf)
        # create budget_manager_param_dict for BalancedIncrementalQuantileFilter used by StreamProbabilisticAL
        budget_manager_param_dict = {"utilities": utilities}
        # update the query strategy and budget_manager to calculate the right budget
        call_func(query_strategy.update, candidates=X_cand, queried_indices=sampled_indices,
                  budget_manager_param_dict=budget_manager_param_dict)
        # count the number of queries
        count += len(sampled_indices)
        # add X_cand to X_train
        X_train.append(x_t)
        # add label or missing_label to y_train
        al_label = y_cand if len(sampled_indices) > 0 else clf.missing_label  #MH: In which cases is len(sampled_indices) > 1?
        y_train.append(y_cand if len(sampled_indices) > 0 else clf.missing_label)

        prediction = clf.predict(X_cand)[0]  #MH: You could already do this before line 51, no?

        if approach_name.endswith('Batch'):
            clf.fit(X_train, y_train)  #MH: Here you use y_train, in line 78 you use al_label. Why?
        elif approach_name.startswith('Clustering'):
            clf.partial_fit(X_cand.reshape([1, -1]), np.array([al_label]))  #MH: (1) Not sure, do you have to reshape again? (probably yes) (2) What happens when we fit / partial_fit on missing labels? couldn't we simply skip it in this case?
        else:
            if not al_label is clf.missing_label:
                clf.partial_fit(X_cand.reshape([1, -1]), np.array([al_label]))

        logger.track_timestep(t)
        logger.track_y(prediction)
        logger.track_label(al_label)
        logger.track_gt(y_cand)
        logger.track_budget(np.round(budget, 1))
        logger.track_bandwidth(band_width)
        logger.track_x1(X_cand[0][0])
        logger.track_x2(X_cand[0][1])
        logger.track_rep(rep)
        tmp_accuracy.append(prediction == y_cand)
        # classification_logger.track_accuracy(sum(accuracy) / len(accuracy))
        logger.track_classifier(approach_name)
        logger.finalize_round()

    df = logger.get_dataframe()
    accuracy = gaussian_filter1d(np.array(tmp_accuracy, dtype=float), 100) #MH: here you could do something like pd.Series(temp_accuracy).rolling(windowsize).mean()
    df["Accuracy"] = accuracy

    # calculate and show the average accuracy
    print("Repition", rep, "Query Strategy: ", approach_name, "Budget: ", budget, "Bandwidth: " , band_width, ", Avg Accuracy: ", np.mean(correct_classifications),
          ", Acquisition count:", count)

    return df


def run_multiple(query_strategies: dict, X, y, logger, n_training_size=0, n_init_traing=10, rep=0, bandwidth=0.1, fit_clf=False):

    results = [run(X, y, query_strategy_name, query_strategy, clf, logger, n_training_size, n_init_traing, rep, bandwidth)
               for query_strategy_name, (query_strategy, clf) in query_strategies.items()]
    return pd.concat(results)
