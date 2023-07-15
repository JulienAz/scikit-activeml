import os

import numpy as np
import pandas as pd
import seaborn as sb
from river.drift.binary import DDM

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ENTROPY, N_CLUSTER, \
    CHANGE_DETECTION, ACCURACY, DETECTOR_THRESHOLD
from skactiveml.stream.clustering.test.mining.util import get_entropy_per_timestep, add_change_dection_to_entropy_df
from skactiveml.stream.clustering.test.parallel_approaches_test import TIMESTEP, BUDGET, REP, CLASSIFIER
from skactiveml.stream.clustering.util import save_image

if __name__ == '__main__':
    subset = -1
    change_detector = DDM

    change_detection_in_log = True

    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "..", "..", target_directory, 'cluster_statistics_time_window.csv')
    csv_acc_filepath = os.path.join(this_dir, "..", "..", target_directory, 'accuracy_time_window.csv')

    df_stat = pd.read_csv(csv_filepath)
    df_acc = pd.read_csv(csv_acc_filepath)

    if subset > 0:
        df_stat = df_stat.loc[df_stat[TIMESTEP] < subset]

    budget_to_plot = np.random.choice(np.unique(df_stat[BUDGET]))

    threshold_to_plot = np.random.choice(np.unique(df_acc[DETECTOR_THRESHOLD]))

    df_acc = df_acc.loc[df_acc[DETECTOR_THRESHOLD] == threshold_to_plot]
    df_stat = df_stat.loc[df_stat[DETECTOR_THRESHOLD] == threshold_to_plot]

    df_stat = df_stat.loc[df_stat[BUDGET] == budget_to_plot]
    df_stat = df_stat.loc[df_stat[CLASSIFIER] == 'ClusteringClfEnsemble']

    df_acc = df_acc.loc[df_acc[BUDGET] == budget_to_plot]
    df_acc = df_acc.loc[df_acc[CLASSIFIER] == 'ClusteringClfEnsemble']

    df_stat = get_entropy_per_timestep(df_stat)
    #df_entropy = add_change_dection_to_entropy_df(df_entropy, change_detector)

    sb.set_theme()

    reps = np.unique(df_stat[REP])

    for r in reps:
        df_plot = df_stat.loc[df_stat[REP] == r]
        df_plot_acc = df_acc.loc[df_acc[REP] == r]

        f = sb.relplot(data=df_plot_acc, x=TIMESTEP, y=ACCURACY,
                       errorbar=None, kind="line", hue='Cluster',
                       col=N_CLUSTER, row=CLASSIFIER,
                       palette='tab10', facet_kws={'sharey': False})
        f.fig.suptitle('Rep: ' + str(r))
        g = sb.relplot(data=df_plot, x=TIMESTEP, y=CHANGE_DETECTION,
                       errorbar=None, kind="line", hue='Cluster',
                       col=N_CLUSTER, row=CLASSIFIER,
                       palette='tab10', facet_kws={'sharey': False})


    image_filepath = os.path.join(this_dir, "..", "..", target_directory, 'output_change_detection_error_per_cluster.pdf')

    save_image(image_filepath)
