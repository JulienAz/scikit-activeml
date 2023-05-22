import numpy as np
from skmultiflow.data import HyperplaneGenerator

from skactiveml.stream.datagenerator import OpenMlStreamGenerator

ABALONE_BIN = 720
ABALONE = 44956
COVERTYPE = 1596
HYPERPLANE = -1


def generate_data(dataset_id, init_train_length, shuffle, random_state, n_features=0, stream_length=None):
    if dataset_id == HYPERPLANE:
        assert n_features > 0, "Please specify the number of features for the hyperplane generator"
        dataGenerator = HyperplaneGenerator(random_state=random_state,
                                            n_features=2,
                                            mag_change=0)
        assert stream_length != None, "Please specify the stream length for hyperplane generator"
    else:
        rng = np.random.default_rng(random_state)
        dataGenerator = OpenMlStreamGenerator(dataset_id, shuffle=shuffle, rng=rng)
    # Generating Datastream
    if stream_length == None:
        stream_length = len(dataGenerator.y) - init_train_length - 1
    X, y = dataGenerator.next_sample(stream_length + init_train_length)
    return X, y
