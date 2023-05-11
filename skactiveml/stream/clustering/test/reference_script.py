import numpy as np
import openml
import sklearn.datasets
import matplotlib.pyplot as plt
from collections import deque
from scipy.ndimage import gaussian_filter1d
from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.stream import StreamRandomSampling, PeriodicSampling
from skactiveml.stream import (
    VariableUncertainty,
    Split,
    StreamProbabilisticAL,
)
from skactiveml.stream.datagenerator._stream_generator import OpenMlStreamGenerator
from skactiveml.utils import call_func

# number of instances that are provided to the classifier
init_train_length = 10
# the length of the data stream
stream_length = 5000
# the size of the sliding window that limits the training data
training_size = 1000
# the parameter dedicated to decide if the classifier needs to be refited with X and y.
fit_clf = False

budget=0.5

# random state that is used to generate random seeds
random_state = np.random.RandomState(0)

def get_randomseed(random_state):
    return random_state.randint(2**31-1)


# Abalone
#dataset = openml.datasets.get_dataset(44956)
#dataGenerator = OpenMlStreamGenerator(dataset)
#stream_length = len(dataGenerator.y) - init_train_length - 1
#X, y = dataGenerator.next_sample(stream_length + init_train_length)


X, y_centers = sklearn.datasets.make_blobs(
    n_samples=init_train_length + stream_length,
    centers=30,
    random_state=get_randomseed(random_state),
    shuffle=True)
y = y_centers % 2

classes = np.unique(y)

X_init = X[:init_train_length, :]
y_init = y[:init_train_length]
X_stream = X[init_train_length:, :]
y_stream = y[init_train_length:]

clf_factory = lambda: ParzenWindowClassifier(classes=classes, random_state=get_randomseed(random_state))
query_strategies = {
    #'StreamRandomSampling': StreamRandomSampling(random_state=get_randomseed(random_state)),
    #'PeriodicSampling': PeriodicSampling(random_state=get_randomseed(random_state)),
    #'VariableUncertainty': VariableUncertainty(random_state=get_randomseed(random_state)),
    #'Split': Split(random_state=get_randomseed(random_state)),
    'StreamProbabilisticAL': StreamProbabilisticAL(random_state=get_randomseed(random_state))
}

for query_strategy_name, query_strategy in query_strategies.items():
    clf = clf_factory()
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
        # train the classifier
        clf.fit(X_train, y_train)
        correct_classifications.append(clf.predict(X_cand)[0] == y_cand)
        # check whether to sample the instance or not
        # call_func is used since a classifier is not needed for RandomSampling and PeriodicSampling
        sampled_indices, utilities = call_func(query_strategy.query, candidates=X_cand, clf=clf, return_utilities=True, fit_clf=fit_clf)
        # create budget_manager_param_dict for BalancedIncrementalQuantileFilter used by StreamProbabilisticAL
        budget_manager_param_dict = {"utilities": utilities}
        # update the query strategy and budget_manager to calculate the right budget
        call_func(query_strategy.update, candidates=X_cand, queried_indices=sampled_indices, budget_manager_param_dict=budget_manager_param_dict)
        # count the number of queries
        count += len(sampled_indices)
        # add X_cand to X_train
        X_train.append(x_t)
        # add label or missing_label to y_train
        y_train.append(y_cand if len(sampled_indices) > 0 else clf.missing_label)
    # calculate and show the average accuracy
    print("Query Strategy: ", query_strategy_name, ", Avg Accuracy: ", np.mean(correct_classifications), ", Acquisition count:", count)
    plt.plot(gaussian_filter1d(np.array(correct_classifications, dtype=float), 50), label=query_strategy_name)
plt.legend()