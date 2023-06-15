import ast

import numpy as np
from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, CLU_TIMEWINDOW, TIMESTEP, RADI, N_SAMPLES, N_CLASSES, N_CLUSTER
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os

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

    # Plotting Cluster Radi over time
    df_radi = df[[REP, BUDGET, TIMESTEP, RADI, CLU_TIMEWINDOW, N_CLUSTER]].copy()
    df_radi = df_radi.loc[df_radi[BUDGET] == budget_to_plot]

    df_radi[RADI] = df_radi[RADI].apply(ast.literal_eval)
    df_radi['RADI_index'] = df_radi.apply(lambda row: [(value, index) for index, value in enumerate(row[RADI])], axis=1)
    df_radi = df_radi.explode('RADI_index').reset_index(drop=True)
    df_radi[[RADI, 'index']] = pd.DataFrame(df_radi['RADI_index'].tolist(), index=df_radi.index)

    sb.set_theme()

    f = sb.relplot(data=df_radi, x=TIMESTEP, y=RADI,
                   errorbar=None, kind="line", hue='index', col=N_CLUSTER,
                   palette='tab10', facet_kws={'sharey': False})

    # Plotting Cluster Number of Samples (n) over time
    df_samples = df[[REP, BUDGET, TIMESTEP, N_SAMPLES, CLU_TIMEWINDOW, N_CLUSTER]].copy()
    df_samples = df_samples.loc[df_samples[BUDGET] == budget_to_plot]

    df_samples[N_SAMPLES] = df_samples[N_SAMPLES].apply(ast.literal_eval)
    df_samples['N_SAMPLE_index'] = df_samples.apply(
        lambda row: [(value, index) for index, value in enumerate(row[N_SAMPLES])], axis=1)
    df_samples = df_samples.explode('N_SAMPLE_index').reset_index(drop=True)
    df_samples[[N_SAMPLES, 'index']] = pd.DataFrame(df_samples['N_SAMPLE_index'].tolist(), index=df_samples.index)

    g = sb.relplot(data=df_samples, x=TIMESTEP, y=N_SAMPLES, errorbar=None, kind="line", hue='index',
                   col=N_CLUSTER, palette='tab10', facet_kws={'sharey': False})

    # Plotting Cluster Number of Labeled Samples over time
    df_labels = df[[REP, BUDGET, TIMESTEP, N_CLASSES, N_CLUSTER]].copy()
    df_labels[N_CLASSES] = df_labels[N_CLASSES].apply(ast.literal_eval)
    df_labels['N_CLASSES_index'] = df_labels.apply(
        lambda row: [(value, index) for index, value in enumerate(row[N_CLASSES])], axis=1)
    df_labels = df_labels.explode('N_CLASSES_index').reset_index(drop=True)
    df_labels[[N_CLASSES, 'index']] = pd.DataFrame(df_labels['N_CLASSES_index'].tolist(), index=df_labels.index)

    h = sb.relplot(data=df_labels, x=TIMESTEP, y=N_CLASSES, errorbar=None, kind="line", hue='index',
                   col=N_CLUSTER    , row=BUDGET, palette='tab10', facet_kws={'sharey': False})

    image_filepath = os.path.join(this_dir, "..", "..", target_directory, 'output_clu_statistics.pdf')

    save_image(image_filepath)
