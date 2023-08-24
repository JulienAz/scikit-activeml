import numpy as np
import openml
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from skmultiflow.data import RandomRBFGenerator


class StreamGenerator:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.index = 0

    def next_sample(self, length):
        if self.index + length > self.X.shape[0]:
            raise ValueError('length exceeds number of remaining samples')

        X_sample = self.X[self.index:self.index + length]
        y_sample = self.y[self.index:self.index + length]

        self.index += length

        return X_sample, y_sample

    def kernel_pca_transformation(self, X):
        kernel_pca = KernelPCA(
            n_components=10, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1
        )

        X = kernel_pca.fit_transform(X)

        return X


class OpenMlStreamGenerator(StreamGenerator):
    def __init__(self, datasetId, rng, shuffle: bool, stream_length, start_point=0):
        dataset = openml.datasets.get_dataset(datasetId)
        self.rng = rng

        # Extract feature matrix and target array
        X, y, categorical_features, _ = dataset.get_data(target=dataset.default_target_attribute)

        if stream_length is not None:
            assert stream_length <= len(y), "Configured stream length exceeds provided OpenML-Dataset length"
            X = X[start_point:start_point + stream_length]
            y = y[start_point:start_point + stream_length]

        # random shuffle of data
        if shuffle:
            indices = np.arange(len(y))
            rng.shuffle(indices)
            X = X.iloc[indices]
            y = y.iloc[indices]

        # Identify numerical features
        numerical_features = ~np.array(categorical_features)

        # Scale numerical features to [0,1]
        numeric_transformer = MinMaxScaler()
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', OneHotEncoder(drop='if_binary'), categorical_features)
            ])

        X = preprocessor.fit_transform(X)

        if isinstance(X, csr_matrix):
            X = X.toarray()
        # Transform y values to integers if categorical
        if y.dtype.name == 'category':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        super().__init__(X, y)


class ArtificialStreamGenerator(StreamGenerator):
    def __init__(self, datasetId, rng, shuffle: bool, stream_length, noise):
        assert stream_length is not None, "Stream length must be provided with artificial strteam generators"
        dataset = openml.datasets.get_dataset(datasetId)
        self.rng = rng

        # Extract feature matrix and target array
        X, y, categorical_features, _ = dataset.get_data(target=dataset.default_target_attribute)

        # random shuffle of data
        if shuffle:
            indices = np.arange(len(y))
            rng.shuffle(indices)
            X = X.iloc[indices]
            y = y.iloc[indices]

        # Identify numerical features
        numerical_features = ~np.array(categorical_features)

        # Scale numerical features to [0,1]
        numeric_transformer = MinMaxScaler()
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])

        X = preprocessor.fit_transform(X)

        # Transform y values to integers if categorical
        if y.dtype.name == 'category':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        indices = rng.choice(len(X), size=stream_length,
                             replace=True)  # Replace 'num_samples' with the desired number of samples to pick
        X = X[indices]
        y = y[indices]

        noise = rng.normal(loc=0, scale=noise,
                           size=X.shape)  # Replace 'noise_std' with the desired standard deviation of the noise
        X = X + noise

        super().__init__(X, y)


class CsvStreamGenerator(StreamGenerator):
    def __init__(self, path, rng, shuffle: bool, stream_length, start_point=0, sample=None):
        dataset = pd.read_csv(path)
        self.rng = rng

        # Extract feature matrix and target array
        target_column = 'target' if 'target' in dataset.columns else 'class'
        y = dataset[target_column]
        X = dataset.drop(columns=target_column)

        # random shuffle of data
        if shuffle:
            indices = np.arange(len(y))
            rng.shuffle(indices)
            X = X.iloc[indices]
            y = y.iloc[indices]

        X = X.values
        y = y.values

        #if stream_length is not None:
        #    if sample is not None:
        #        X = X[start_point: start_point + sample * stream_length: sample]
        #        y = y[start_point: start_point + sample * stream_length: sample]
        #    else:
        #        X = X[start_point:stream_length + start_point, :]
        #        y = y[start_point:stream_length + start_point]
        X = X[start_point:stream_length + start_point, :]
        y = y[start_point:stream_length + start_point]
        super().__init__(X, y)


class RbfStreamGenerator(StreamGenerator):

    def _distance(self, point_a, point_b):
        return np.linalg.norm(np.array(point_a) - np.array(point_b))

    def __init__(self, random_state, stream_length, n_features=2, n_classes=5, n_centroids=5, stdv_min=0.005, stdv_max=0.1, min_dist=0.01):
        assert n_classes == n_centroids

        # Extract feature matrix and target array
        stream_gen = RandomRBFGenerator(model_random_state=random_state, n_classes=n_classes, n_features=n_features, n_centroids=n_centroids)
        rng = np.random.default_rng(random_state)

        X, y = np.empty((0, n_features)), np.empty((0), dtype=int)

        for i in range(n_classes):
            stream_gen.centroids[i].class_label = i
            stream_gen.centroids[i].std_dev = rng.uniform(stdv_min, stdv_max)

        for i in range(n_centroids):
            for j in range(i + 1, n_centroids):
                dist = self._distance(stream_gen.centroids[i].centre, stream_gen.centroids[j].centre)
                while dist < min_dist:
                    # Move one of the centroids
                    stream_gen.centroids[j].centre = rng.uniform(0, 1, size=n_features)
                    dist = self._distance(stream_gen.centroids[i].centre, stream_gen.centroids[j].centre)

        for i in range(int(stream_length / 2000)):
            # Generate next 2000 samples
            X_tmp, y_tmp = stream_gen.next_sample(2000)
            X = np.concatenate((X, X_tmp))
            y = np.concatenate((y, y_tmp))

            # Randomly choose number of rbfs pairs to swap
            n_pairs_to_swap = rng.integers(1, len(stream_gen.centroids) // 2)

            # Randomly select the RBFs to swap
            rbfs_to_swap = rng.choice(stream_gen.centroids, n_pairs_to_swap * 2, replace=False)

            # Swap the attributes for the selected RBF pairs
            for i in range(0, len(rbfs_to_swap), 2):
                rbfs_to_swap[i].class_label, rbfs_to_swap[i + 1].class_label = rbfs_to_swap[i + 1].class_label, rbfs_to_swap[i].class_label
                rbfs_to_swap[i].std_dev, rbfs_to_swap[i + 1].std_dev = rbfs_to_swap[i + 1].std_dev, rbfs_to_swap[i].std_dev


        rest_length = stream_length % 2000
        X_tmp, y_tmp = stream_gen.next_sample(rest_length)
        X = np.concatenate((X, X_tmp))
        y = np.concatenate((y, y_tmp))

        super().__init__(X, y)
