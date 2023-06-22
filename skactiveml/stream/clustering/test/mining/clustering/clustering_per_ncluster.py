import numpy as np
from matplotlib import pyplot as plt, patches

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, TIMESTEP, LABEL_FREQUENCY, CLUSTER, GT, Y, X1, X2, C1, C2, RADIUS, \
    CLU_TIMEWINDOW, N_CLUSTER
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os

if __name__ == '__main__':

    n_reps_to_plot = 3
    rep = 1

    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'

    csv_clustering_filepath = os.path.join(this_dir, "..", "..", target_directory, 'clustering_time_window.csv')
    df_clustering = pd.read_csv(csv_clustering_filepath)

    csv_clustering_filepath = os.path.join(this_dir, "..", "..", target_directory, 'accuracy_time_window.csv')
    df_acc = pd.read_csv(csv_clustering_filepath)

    #df_clustering = df_clustering.loc[df_clustering[REP] == rep]
    #df_acc = df_acc.loc[df_acc[REP] == rep]

    #df_clustering = df_clustering.loc[df_clustering[CLASSIFIER] == 'ClusteringClfRefit']
    #df_acc = df_acc.loc[df_acc[CLASSIFIER] == 'ClusteringClfRefit']

    budget_to_plot = np.random.choice(np.unique(df_acc[BUDGET]))
    df_clustering = df_clustering.loc[df_clustering[BUDGET] == budget_to_plot]
    df_acc = df_acc.loc[df_acc[BUDGET] == budget_to_plot]

    df_merged = pd.merge(df_clustering, df_acc, on=[X1, X2, REP, CLASSIFIER, BUDGET, CLU_TIMEWINDOW, N_CLUSTER])

    sb.set_theme()

    g = sb.relplot(data=df_merged, x=X1, y=X2, kind='scatter',
                   palette="deep", edgecolor='White', linewidth=0.5,
                   hue=GT, col=N_CLUSTER, row=REP)
    df_labeled = df_merged.loc[~np.isnan(df_merged['Label_x'])]

    axes = g.axes

    center_radi = df_clustering[[C1, C2, RADIUS, REP, CLASSIFIER, BUDGET, N_CLUSTER]].drop_duplicates().to_numpy()
    # Sort the array by column Buget and then column Timewindow
    sorted_indices = np.lexsort((center_radi[:, 6], center_radi[:, 5], center_radi[:, 4], center_radi[:, 3]))
    center_radi = center_radi[sorted_indices]
    # Add Cluster Radi
    n_budget = 1
    n_cluster_values = np.unique(df_merged[N_CLUSTER])
    n_reps = len(np.unique(df_merged[REP]))

    index = 0

    for r in range(n_reps):
        for t, n_cluster in enumerate(n_cluster_values):
            for c in range(n_cluster):
                c1 = center_radi[index][0]
                c2 = center_radi[index][1]
                radi = center_radi[index][2]
                axes[r][t].add_patch(patches.Circle([c1, c2], radius=radi * 3, fill=False, color="turquoise"))
                index += 1

    image_filepath = os.path.join(this_dir, "..", "..", target_directory, 'output_clustering_per_ncluster.pdf')
    save_image(image_filepath)

