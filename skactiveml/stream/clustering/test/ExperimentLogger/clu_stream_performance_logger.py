from datetime import time

import numpy as np
import pandas as pd

REP = "rep"
TIMESTEP = "round"
APPROACH = "Approach"
ACCURACY = "Accuracy"
N_CLUSTER = "n_Cluster"
CLUSTER = "Cluster"
DATASET = "Dataset"
CLASSIFIER = "Classifier"
STRATEGY = "LabelingStrategy"
CHANGE_DETECTOR = "ChangeDetector"
TIME = "time"
BUDGET = "Budget"
BANDWIDTH = "Kernel_Bandwidth"
X1 = "X1"
X2 = "X2"
Y = "Prediction"
LABEL = "Label"
GT = "GT_Label"
CLU_TIMEWINDOW = "CluStreamTimewindow"
LABEL_FREQUENCY = "Lbl_frequency"
C1 = "C1"
C2 = "C2"
RADIUS = "Radius"
CENTERS = "Centers"
RADI = "Radi"
N_CLASSES = "N_Classes"
CLASS_DIST = "Class_Distribution"
N_SAMPLES = "N_Samples"
LS_X = "LS_x"
ENTROPY = "Class_Entropy"
CHANGE_DETECTION = "Change_Detection"
DETECTOR_THRESHOLD = 'DetectionThreshold'
W = 'w'

# Add column names to list
ids_acc = [
    REP,
    TIMESTEP,
    ACCURACY,
    N_CLUSTER,
    CLUSTER,
    DATASET,
    CLASSIFIER,
    STRATEGY,
    CHANGE_DETECTOR,
    BUDGET,
    BANDWIDTH,
    X1,
    X2,
    Y,
    LABEL,
    GT,
    CLU_TIMEWINDOW,
    LABEL_FREQUENCY,
    DETECTOR_THRESHOLD,
    W
]


class CluStreamPerformanceLogger:

    # Nothing to do here
    def __init__(self):
        self._columns = ids_acc
        self._data = []
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._column_indices = {key: i for (i, key) in enumerate(self._columns)}

    # Add own functions for tracking different metrics
    def track_rep(self, value: int):
        self._track_value(value, REP)

    def track_dataset(self, value: str):
        self._track_value(value, DATASET)

    def track_classifier(self, value: str):
        self._track_value(value, CLASSIFIER)

    def track_change_detector(self, value:str):
        self._track_value(value, CHANGE_DETECTOR)

    def track_strategy(self, value:str):
        self._track_value(value, STRATEGY)

    def track_accuracy(self, value: float):
        self._track_value(value, ACCURACY)

    def track_timestep(self, value: int):
        self._track_value(value, TIMESTEP)

    def track_budget(self, value: int):
        self._track_value(value, BUDGET)

    def track_bandwidth(self, value: float):
        self._track_value(value, BANDWIDTH)

    def track_n_cluster(self, value: int):
        self._track_value(value, N_CLUSTER)

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

    def track_clu_time_window(self, value):
        self._track_value(value, CLU_TIMEWINDOW)

    def track_detector_threshold(self, value):
        self._track_value(value, DETECTOR_THRESHOLD)

    def track_w(self, value):
        self._track_value(value, W)

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


ids_clustering = [
    REP,
    CLUSTER,
    DATASET,
    CLASSIFIER,
    STRATEGY,
    CHANGE_DETECTOR,
    BUDGET,
    BANDWIDTH,
    N_CLUSTER,
    X1,
    X2,
    LABEL,
    C1,
    C2,
    RADIUS,
    CLU_TIMEWINDOW
]


class CluStreamClusteringLogger:
    def __init__(self):
        self._columns = ids_clustering
        self._data = []
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._column_indices = {key: i for (i, key) in enumerate(self._columns)}

    # Add own functions for tracking different metrics
    def track_rep(self, value: int):
        self._track_value(value, REP)

    def track_dataset(self, value: str):
        self._track_value(value, DATASET)

    def track_classifier(self, value: str):
        self._track_value(value, CLASSIFIER)

    def track_change_detector(self, value: str):
        self._track_value(value, CHANGE_DETECTOR)

    def track_strategy(self, value: str):
        self._track_value(value, STRATEGY)

    def track_budget(self, value: int):
        self._track_value(value, BUDGET)

    def track_bandwidth(self, value: float):
        self._track_value(value, BANDWIDTH)

    def track_n_cluster(self, value: int):
        self._track_value(value, N_CLUSTER)

    def track_cluster(self, value: int):
        self._track_value(value, CLUSTER)

    def track_x1(self, value):
        self._track_value(value, X1)

    def track_x2(self, value):
        self._track_value(value, X2)

    def track_label(self, value):
        self._track_value(value, LABEL)

    def track_clu_time_window(self, value):
        self._track_value(value, CLU_TIMEWINDOW)

    def track_cluster_c1(self, value):
        self._track_value(value, C1)

    def track_cluster_c2(self, value):
        self._track_value(value, C2)

    def track_cluster_radi(self, value):
        self._track_value(value, RADIUS)

    def finalize_round(self):
        self._data.append(self._current_row)
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._data, columns=self._columns)
        self._reset()
        return df

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

ids_cluster_statistics = [
    TIMESTEP,
    REP,
    DATASET,
    CLASSIFIER,
    BUDGET,
    BANDWIDTH,
    CLU_TIMEWINDOW,
    N_CLASSES,
    N_CLUSTER,
    CLASS_DIST,
    CENTERS,
    RADI,
    N_SAMPLES,
    LS_X,
    ENTROPY,
    CHANGE_DETECTION,
    DETECTOR_THRESHOLD
]


class CluStreamStatisticLogger:
    def __init__(self):
        self._columns = ids_cluster_statistics
        self._data = []
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._column_indices = {key: i for (i, key) in enumerate(self._columns)}

    # Add own functions for tracking different metrics
    def track_timestep(self, value: int):
        self._track_value(value, TIMESTEP)

    def track_rep(self, value: int):
        self._track_value(value, REP)

    def track_dataset(self, value: str):
        self._track_value(value, DATASET)

    def track_classifier(self, value: str):
        self._track_value(value, CLASSIFIER)

    def track_budget(self, value: int):
        self._track_value(value, BUDGET)

    def track_bandwidth(self, value: float):
        self._track_value(value, BANDWIDTH)

    def track_n_cluster(self, value: int):
        self._track_value(value, N_CLUSTER)

    def track_clu_time_window(self, value):
        self._track_value(value, CLU_TIMEWINDOW)

    def track_cluster_centers(self, value):
        self._track_value(value, CENTERS)

    def track_cluster_radi(self, value):
        self._track_value(value, RADI)

    def track_n_classes(self, value):
        self._track_value(value, N_CLASSES)

    def track_classes_dist(self, value):
        self._track_value(value, CLASS_DIST)

    def track_n_samples(self, value):
        self._track_value(value, N_SAMPLES)

    def track_ls_x(self, value):
        self._track_value(value, LS_X)

    def track_class_entropy(self, value):
        self._track_value(value, ENTROPY)

    def track_change_detection(self, value):
        self._track_value(value, CHANGE_DETECTION)

    def track_detector_threshold(self, value):
        self._track_value(value, DETECTOR_THRESHOLD)

    def finalize_round(self):
        self._data.append(self._current_row)
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._data, columns=self._columns)
        self._reset()
        return df

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


ids_change_detection = [
    TIMESTEP,
    REP,
    DATASET,
    CLASSIFIER,
    BUDGET,
    CHANGE_DETECTION,
]

class DetectionLogger:
    def __init__(self):
        self._columns = ids_change_detection
        self._data = []
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._column_indices = {key: i for (i, key) in enumerate(self._columns)}

    # Add own functions for tracking different metrics
    def track_timestep(self, value: int):
        self._track_value(value, TIMESTEP)

    def track_rep(self, value: int):
        self._track_value(value, REP)

    def track_dataset(self, value: str):
        self._track_value(value, DATASET)

    def track_classifier(self, value: str):
        self._track_value(value, CLASSIFIER)

    def track_budget(self, value: int):
        self._track_value(value, BUDGET)

    def track_change_detection(self, value):
        self._track_value(value, CHANGE_DETECTION)

    def finalize_round(self):
        self._data.append(self._current_row)
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._data, columns=self._columns)
        self._reset()
        return df

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

ids_adaption = [
    REP,
    X1,
    X2,
    LABEL,
    DATASET,
    CLASSIFIER,
    BUDGET,
]

class AdaptionLogger:
    def __init__(self):
        self._columns = ids_adaption
        self._data = []
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._column_indices = {key: i for (i, key) in enumerate(self._columns)}

    # Add own functions for tracking different metrics
    def track_timestep(self, value: int):
        self._track_value(value, TIMESTEP)

    def track_rep(self, value: int):
        self._track_value(value, REP)

    def track_dataset(self, value: str):
        self._track_value(value, DATASET)

    def track_classifier(self, value: str):
        self._track_value(value, CLASSIFIER)

    def track_budget(self, value: int):
        self._track_value(value, BUDGET)

    def track_x1(self, value):
        self._track_value(value, X1)

    def track_x2(self, value):
        self._track_value(value, X2)

    def track_label(self, value):
        self._track_value(value, LABEL)

    def finalize_round(self):
        self._data.append(self._current_row)
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._data, columns=self._columns)
        self._reset()
        return df

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