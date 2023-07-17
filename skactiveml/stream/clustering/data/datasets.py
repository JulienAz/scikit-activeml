import os

import numpy as np
import openml
import pandas as pd
from skmultiflow.data import HyperplaneGenerator

from skactiveml.stream.datagenerator import OpenMlStreamGenerator, ArtificialStreamGenerator, CsvStreamGenerator
from skactiveml.stream.datagenerator._stream_generator import RbfStreamGenerator

# Open ML datasets
ABALONE_BIN = {'name': 'Abalone_binary', 'type': 'openml', 'id': 720, 'toy': False}
ABALONE = {'name': 'Abalone', 'type': 'openml', 'id': 44956, 'toy': False}
COVERTYPE = {'name': 'Covertype', 'length': 581012, 'type': 'openml', 'id': 1596, 'toy': False}
IRIS = {'name': 'Abalone_binary', 'type': 'openml', 'id': 61, 'toy': True}

# ScikitMultiflow
HYPERPLANE = {'name': 'Hyperplane', 'type': 'hyperplane'}
RBF_GENERATOR = {'name': 'RbfGenerator', 'type': 'rbf'}

# Local csv sets
# Real World
AIRLINES = {'name': 'Airlines', 'length': 539383, 'type': 'local', 'path': os.path.join('real-world', 'airlines2.csv')}
ELECTRICITY = {'name': 'Electricity', 'length': 45312, 'type': 'local', 'path': os.path.join('real-world', 'elec.csv')}

# Artificial
CHESSBOARD = {'name': 'ChessBoard', 'length': 200000, 'type': 'local', 'path': os.path.join('artificial', 'transientChessboard.csv')}
INTERCHANGING_RBF = {'name': 'InterchangingRBF', 'length': 200000, 'type': 'local', 'path': os.path.join('artificial', 'interchangingRBF.csv')}
ROTATING_HYPERPLANE = {'name': 'RotatingHyperplane', 'type': 'local', 'path': os.path.join('artificial', 'rotatingHyperplane.csv')}
SEA_STREAM = {'name': 'SeaStream', 'type': 'local', 'path': os.path.join('artificial', 'sea_stream.csv')}


def generate_data(dataset, init_train_length, shuffle, random_state, n_features=0, stream_length=4000, mag_change=0,
                  noise=0.1, start_point=0, kpca=False):
    set_type = dataset['type']
    if set_type == 'hyperplane':
        assert n_features > 0, "Please specify the number of features for the hyperplane generator"
        dataGenerator = HyperplaneGenerator(random_state=random_state,
                                            n_features=n_features,
                                            mag_change=mag_change)
        assert stream_length != None, "Please specify the stream length for hyperplane generator"
    elif set_type == 'rbf':
        dataGenerator = RbfStreamGenerator(random_state=random_state,
                                           stream_length=stream_length + init_train_length,
                                           )
        assert stream_length != None, "Please specify the stream length for hyperplane generator"
    elif set_type == 'openml':
        if dataset['toy']:
            rng = np.random.default_rng(random_state)
            dataGenerator = ArtificialStreamGenerator(dataset['id'], shuffle=shuffle, rng=rng,
                                                      stream_length=stream_length + init_train_length, noise=noise)
        else:
            rng = np.random.default_rng(random_state)
            dataGenerator = OpenMlStreamGenerator(dataset['id'], shuffle=shuffle, rng=rng,
                                                  stream_length=stream_length + init_train_length, start_point=start_point)
    elif set_type == 'local':
        rng = np.random.default_rng(random_state)
        path = os.path.join(os.path.dirname(__file__), dataset['path'])
        dataGenerator = CsvStreamGenerator(path, shuffle=shuffle, rng=rng, stream_length=stream_length + init_train_length, start_point=start_point)
    # Generating Datastream
    if stream_length is None:
        stream_length = len(dataGenerator.y) - init_train_length - 1

    X, y = dataGenerator.next_sample(stream_length + init_train_length)

    if kpca:
        X = dataGenerator.kernel_pca_transformation(X)

    return X, y
