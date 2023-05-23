from datetime import time

import numpy as np
import pandas as pd

REP = "rep"
TIMESTEP = "round"
APPROACH = "Approach"
ACCURACY = "Accuracy"
CLUSTER = "Cluster"
CLASSIFIER = "Classifier"
TIME = "time"
BUDGET = "Budget"
BANDWIDTH = "Kernel_Bandwidth"
X1 = "X1"
X2 = "X2"
Y = "Prediction"
LABEL = "Label"
GT = "GT_Label"
LABEL_FREQUENCY = "Lbl_frequency"

# Add column names to list
all_ids = [
    REP,
    TIMESTEP,
    ACCURACY,
    CLUSTER,
    CLASSIFIER,
    BUDGET,
    BANDWIDTH,
    X1,
    X2,
    Y,
    LABEL,
    GT,
    LABEL_FREQUENCY
]


class CluStreamPerformanceLogger:

    # Nothing to do here
    def __init__(self):
        self._columns = all_ids
        self._data = []
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._column_indices = {key: i for (i, key) in enumerate(self._columns)}

    # Add own functions for tracking different metrics
    def track_rep(self, value: int):
        self._track_value(value, REP)

    def track_classifier(self, value: str):
        self._track_value(value, CLASSIFIER)

    def track_accuracy(self, value: float):
        self._track_value(value, ACCURACY)

    def track_timestep(self, value: int):
        self._track_value(value, TIMESTEP)

    def track_budget(self, value: int):
        self._track_value(value, BUDGET)

    def track_bandwidth(self, value: float):
        self._track_value(value, BANDWIDTH)

    def track_cluster(self, value: int):
        self._track_value(value, CLUSTER)

    def track_x1(self, value):
        self._track_value(value, X1)

    def track_x2(self, value):
        self._track_value(value, X2)

    def track_y(self, value):
        self._track_value(value, Y)

    def track_label(self, value):
        self._track_value(value, LABEL)

    def track_gt(self, value):
        self._track_value(value, GT)

    def track_lbl_frequency(self, value):
        self._track_value(value, LABEL_FREQUENCY)

    def finalize_round(self):
        self._data.append(self._current_row)
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._data, columns=self._columns)
        self._reset()
        return df

    # Noting to do here
    def track_time(self):
        current_time = time.perf_counter_ns()
        self._track_value(current_time, "time")

    def _track_value(self, newval, id):
        self._current_row[self._index_of(id)] = newval

    def _index_of(self, id):
        return self._column_indices[id]

    def _reset(self):
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._data = []

    @property
    def data(self):
        return self._data