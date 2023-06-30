import os

import numpy as np
import pandas as pd
import seaborn as sb
from river.drift.binary import DDM

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ENTROPY, N_CLUSTER, \
    CHANGE_DETECTION
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
    df = pd.read_csv(csv_filepath)

    if subset > 0:
        df = df.loc[df[TIMESTEP] < subset]

    budget_to_plot = np.random.choice(np.unique(df[BUDGET]))
    df = df.loc[df[BUDGET] == budget_to_plot]
    df = df.loc[df[CLASSIFIER] == 'ClusteringClfRefit']

    df_entropy = get_entropy_per_timestep(df)
    #df_entropy = add_change_dection_to_entropy_df(df_entropy, change_detector)

    sb.set_theme()

    reps = np.unique(df_entropy[REP])

    for r in reps:
        df_plot = df_entropy.loc[df_entropy[REP] == r]
        f = sb.relplot(data=df_plot, x=TIMESTEP, y=ENTROPY,
                       errorbar=None, kind="line", hue='Cluster',
                       col=N_CLUSTER, row=CLASSIFIER,
                       palette='tab10', facet_kws={'sharey': False})
        f.fig.suptitle('Rep: ' + str(r))
        g = sb.relplot(data=df_plot, x=TIMESTEP, y=CHANGE_DETECTION,
                       errorbar=None, kind="line", hue='Cluster',
                       col=N_CLUSTER, row=CLASSIFIER,
                       palette='tab10', facet_kws={'sharey': False})


    image_filepath = os.path.join(this_dir, "..", "..", target_directory, 'output_change_detection_entropy_per_cluster.pdf')

    save_image(image_filepath)
