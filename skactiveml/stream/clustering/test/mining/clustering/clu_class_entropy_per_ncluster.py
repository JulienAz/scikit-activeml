import ast

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import entropy

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, CLU_TIMEWINDOW, TIMESTEP, RADI, N_SAMPLES, N_CLASSES, CLASS_DIST, N_CLUSTER, \
    ENTROPY
from skactiveml.stream.clustering.test.mining.util import convert_to_array, get_entropy_per_timestep
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os

if __name__ == '__main__':
    rep_to_plot = 0
    budget_to_plot = 0.21
    subset = -1

    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "..", "..", target_directory, 'cluster_statistics_time_window.csv')
    df = pd.read_csv(csv_filepath)

    if subset > 0:
        df = df.loc[df[TIMESTEP] < subset]
    budget_to_plot = np.random.choice(np.unique(df[BUDGET]))
    df = df.loc[df[BUDGET] == budget_to_plot]

    df[CLASS_DIST] = df[CLASS_DIST].apply(convert_to_array)
    df_entropy = get_entropy_per_timestep(df)

    sb.set_theme()

    reps = np.unique(df_entropy[REP])

    for r in reps:
        df_plot = df_entropy.loc[df_entropy[REP] == r]
        f = sb.relplot(data=df_plot, x=TIMESTEP, y=ENTROPY,
                       errorbar=None, kind="line", hue='Cluster',
                       col=N_CLUSTER, row=CLASSIFIER,
                       palette='tab10', facet_kws={'sharey': False})
        f.fig.suptitle('Rep: ' + str(r))


    image_filepath = os.path.join(this_dir, "..", "..", target_directory, 'output_clu_entropy_per_cluster.pdf')

    save_image(image_filepath)
