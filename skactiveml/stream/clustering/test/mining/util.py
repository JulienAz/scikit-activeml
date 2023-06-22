import ast

import numpy as np
import pandas as pd
from river.drift import ADWIN
from scipy.stats import entropy

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ENTROPY, TIMESTEP, \
    CLU_TIMEWINDOW, N_CLUSTER, CLASSIFIER, REP, CLASS_DIST, CLUSTER, CHANGE_DETECTION


def convert_to_array(entry):
    entry = entry.replace("array(", "").replace(")", "")
    return np.array(eval(entry))

# Calculate and add Entropy from logged Class distribution per cluster (If entropy is not logged directly)
def calculate_entropy_per_timestep_from_class_dist(df):
    entropy_per_timestep = []
    n_classes = len(df[CLASS_DIST][0])
    for _, row in df.iterrows():
        timestep = row[TIMESTEP]
        class_distributions = row[CLASS_DIST]
        clu_timewindow = row[CLU_TIMEWINDOW]
        n_cluster = row[N_CLUSTER]
        classifier = row[CLASSIFIER]
        rep = row[REP]
        class_entropies = []
        entropy_test = ast.literal_eval(row[ENTROPY])

        for i, class_distribution in enumerate(class_distributions):
            test = entropy_test[i]
            class_distribution = np.array(class_distribution)
            class_probabilities = class_distribution / np.sum(class_distribution)
            class_entropy = entropy(class_probabilities, base=n_classes)
            class_entropies.append([timestep, rep, clu_timewindow, n_cluster, classifier, i + 1, entropy_test[i]])

        entropy_per_timestep.extend(class_entropies)
        column_names = [TIMESTEP, REP, CLU_TIMEWINDOW, N_CLUSTER, CLASSIFIER, 'Cluster', 'ClassEntropy']

    return pd.DataFrame(entropy_per_timestep, columns=column_names)

# When Entropy is logged directly, convert to numerical and add to df
def get_entropy_per_timestep(df, change_detection=True):
    entropy_per_timestep = []
    df[ENTROPY] = df[ENTROPY].apply(ast.literal_eval)
    if change_detection:
        df[CHANGE_DETECTION] = df[CHANGE_DETECTION].apply(ast.literal_eval)
    for _, row in df.iterrows():
        timestep = row[TIMESTEP]
        clu_timewindow = row[CLU_TIMEWINDOW]
        n_cluster = row[N_CLUSTER]
        classifier = row[CLASSIFIER]
        rep = row[REP]
        class_entropies = []
        entropy_per_cluster = row[ENTROPY]
        detection_per_cluster = row[CHANGE_DETECTION]

        for i, entropy in enumerate(entropy_per_cluster):
            class_entropies.append([timestep, rep, clu_timewindow, n_cluster,
                                    classifier, i, entropy, detection_per_cluster[i]])

        entropy_per_timestep.extend(class_entropies)
        column_names = [TIMESTEP, REP, CLU_TIMEWINDOW, N_CLUSTER, CLASSIFIER, CLUSTER, ENTROPY, CHANGE_DETECTION]

    return pd.DataFrame(entropy_per_timestep, columns=column_names)

def add_change_dection_to_entropy_df(df, detector_logged=True, change_detector=ADWIN):
    reps = np.unique(df[REP])
    cluster_sizes = np.unique(df[N_CLUSTER])
    classifiers = np.unique(df[CLASSIFIER])

    # Init Change detectors
    if detector_logged:
        df[CHANGE_DETECTION] = df[CHANGE_DETECTION].apply(ast.literal_eval)
    else:
        change_detectors = np.array([[[change_detector() for cluster in range(n_cluster)] for n_cluster in cluster_sizes] for c in enumerate(classifiers)], dtype=object)
    change_detection = []
    for _, row in df.iterrows():
        rep = row[REP]
        n_cluster = row[N_CLUSTER]
        cluster = row[CLUSTER]
        n_cluster_rep = np.argmax(cluster_sizes == n_cluster)

        classifier = row[CLASSIFIER]
        classifier_index = np.argmax(classifiers == classifier)

        entropy = row[ENTROPY]

        # Get corresponding detector

        # Update Detector with entropy value
        if detector_logged:
            change_detection.append(row[CHANGE_DETECTION])

        change_detectors[classifier_index, n_cluster_rep][cluster].update(entropy)

        # Log if change detected
        change_detection.append(change_detectors[classifier_index, n_cluster_rep][cluster].drift_detected)
    df[CHANGE_DETECTION] = change_detection
    return df
