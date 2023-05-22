import numpy as np
import openml
import pandas as pd
from openml import OpenMLDataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder


class StreamGenerator:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.index = 0

    def next_sample(self, length):
        if self.index + length > self.X.shape[0]:
            raise ValueError('length exceeds number of remaining samples')

        X_sample = self.X[self.index:self.index+length]
        y_sample = self.y[self.index:self.index+length]

        self.index += length

        return X_sample, y_sample

class OpenMlStreamGenerator(StreamGenerator):
    def __init__(self, datasetId, rng, shuffle: bool):
        dataset = openml.datasets.get_dataset(datasetId)
        self.rng = rng

        # Extract feature matrix and target array
        X, y, categorical_features, _ = dataset.get_data(target=dataset.default_target_attribute)

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
        if isinstance(y[0], str):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # random shuffle of data
        if shuffle:
            indices = np.arange(len(y))
            rng.shuffle(indices)
            X = X[indices]
            y = y[indices]

        super().__init__(X, y)
