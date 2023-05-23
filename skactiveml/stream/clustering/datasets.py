import numpy as np
from skmultiflow.data import HyperplaneGenerator

from skactiveml.stream.datagenerator import OpenMlStreamGenerator, ArtificialStreamGenerator

ABALONE_BIN = 720
ABALONE = 44956
COVERTYPE = 1596
HYPERPLANE = -1
IRIS = 61


def generate_data(dataset_id, init_train_length, shuffle, random_state, n_features=0, stream_length=None, mag_change=0, noise=0.1):
    if dataset_id == HYPERPLANE:
        assert n_features > 0, "Please specify the number of features for the hyperplane generator"
        dataGenerator = HyperplaneGenerator(random_state=random_state,
                                            n_features=2,
                                            mag_change=mag_change)
        assert stream_length != None, "Please specify the stream length for hyperplane generator"
    elif dataset_id == IRIS:
        rng = np.random.default_rng(random_state)
        dataGenerator = ArtificialStreamGenerator(dataset_id, shuffle=shuffle, rng=rng, stream_length=stream_length + init_train_length, noise=noise)
    else:
        rng = np.random.default_rng(random_state)
        dataGenerator = OpenMlStreamGenerator(dataset_id, shuffle=shuffle, rng=rng, stream_length=stream_length + init_train_length)
    # Generating Datastream
    if stream_length is None:
        stream_length = len(dataGenerator.y) - init_train_length - 1

    X, y = dataGenerator.next_sample(stream_length + init_train_length)
    return X, y
