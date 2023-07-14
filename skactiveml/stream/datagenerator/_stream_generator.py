import numpy as np
import openml
import pandas as pd
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
    def __init__(self, datasetId, rng, shuffle: bool, stream_length):
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

        if stream_length is not None:
            assert stream_length <= len(y), "Configured stream length exceeds provided OpenML-Dataset length"
            X = X[:stream_length]
            y = y[:stream_length]

        # Identify numerical features
        numerical_features = ~np.array(categorical_features)

        # Scale numerical features to [0,1]
        numeric_transformer = MinMaxScaler()
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', OneHotEncoder(drop='if_binary'), categorical_features)
            ])

        X = preprocessor.fit_transform(X).toarray()

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
    def __init__(self, path, rng, shuffle: bool, stream_length, start_point=0):
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

        if stream_length is not None:
            X = X[start_point:stream_length + start_point, :]
            y = y[start_point:stream_length + start_point]
        super().__init__(X, y)


class RbfStreamGenerator(StreamGenerator):
    def __init__(self, random_state, stream_length, n_features=2, n_classes=3, n_centroids=3, stdv_max=0.03):
        assert n_classes == n_centroids

        # Extract feature matrix and target array
        stream_gen = RandomRBFGenerator(model_random_state=random_state, n_classes=n_classes, n_features=n_features, n_centroids=n_centroids)
        rng = np.random.default_rng(random_state)
        for i in range(n_classes):
            stream_gen.centroids[i].class_label = i
            stream_gen.centroids[i].std_dev = rng.uniform(0, stdv_max)
        X, y = stream_gen.next_sample(int(stream_length / 2))

        # !!! Hardcoding concept drift by swapping class labels and stdv of RBFs
        stream_gen.centroids[0].class_label = 1
        stream_gen.centroids[1].class_label = 0

        std_dev_0 = stream_gen.centroids[0].std_dev
        std_dev_1 = stream_gen.centroids[1].std_dev
        stream_gen.centroids[0].std_dev = std_dev_1
        stream_gen.centroids[1].std_dev = std_dev_0


        X_tmp, y_tmp = stream_gen.next_sample(int(stream_length / 2) + 1)
        X = np.concatenate((X, X_tmp))
        y = np.concatenate((y, y_tmp))
        super().__init__(X, y)
