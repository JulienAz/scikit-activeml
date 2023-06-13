import ast

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import entropy

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, CLU_TIMEWINDOW, TIMESTEP, RADI, N_SAMPLES, N_CLASSES, CLASS_DIST
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os

def convert_to_array(entry):
    entry = entry.replace("array(", "").replace(")", "")
    return np.array(eval(entry))

def calculate_entropy_per_timestep(df):
    entropy_per_timestep = []

    for _, row in df.iterrows():
        timestep = row[TIMESTEP]
        class_distributions = row[CLASS_DIST]
        class_entropies = []

        for i, class_distribution in enumerate(class_distributions):
            class_distribution = np.array(class_distribution)
            class_probabilities = class_distribution / np.sum(class_distribution)
            class_entropy = entropy(class_probabilities, base=2)
            class_entropies.append([timestep, i + 1, class_entropy])

        entropy_per_timestep.extend(class_entropies)
        column_names = [TIMESTEP, 'Cluster', 'ClassEntropy']

    return pd.DataFrame(entropy_per_timestep, columns=column_names)

if __name__ == '__main__':
    rep_to_plot = 0
    budget_to_plot = 0.21

    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "..", "..", target_directory, 'cluster_statistics_time_window.csv')
    df = pd.read_csv(csv_filepath)

    budget_to_plot = np.random.choice(np.unique(df[BUDGET]))
    df = df.loc[df[REP] == rep_to_plot]
    df = df.loc[df[BUDGET] == budget_to_plot]

    df[CLASS_DIST] = df[CLASS_DIST].apply(convert_to_array)
    df_entropy = calculate_entropy_per_timestep(df)

    sb.set_theme()

    f = sb.relplot(data=df_entropy, x=TIMESTEP, y='ClassEntropy',
                   errorbar=None, kind="line", hue='Cluster',
                   palette='tab10', facet_kws={'sharey': False})


    image_filepath = os.path.join(this_dir, "..", "..", target_directory, 'output_clu_entropy.pdf')

    save_image(image_filepath)
