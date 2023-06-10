import numpy as np
from matplotlib import pyplot as plt, patches

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, TIMESTEP, LABEL_FREQUENCY, CLUSTER, GT, Y, X1, X2, C1, C2, RADIUS, CLU_TIMEWINDOW
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os

def add_points(x, y, dataframe, **kwargs):
    # Access the column and row values of the current subplot
    column_value = dataframe[CLU_TIMEWINDOW]
    row_value = dataframe[BUDGET]

    df_filtered = dataframe.loc[(dataframe[BUDGET] == row_value)& (dataframe[CLU_TIMEWINDOW] == column_value)]
    plt.scatter(df_filtered[x], df_filtered[y], edgecolor='red', s=50)

if __name__ == '__main__':

    n_reps_to_plot = 3
    rep = 0

    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'

    csv_clustering_filepath = os.path.join(this_dir, "..", target_directory, 'clustering_time_window.csv')
    df_clustering = pd.read_csv(csv_clustering_filepath)

    csv_clustering_filepath = os.path.join(this_dir, "..", target_directory, 'accuracy_time_window.csv')
    df_acc = pd.read_csv(csv_clustering_filepath)

    df_clustering = df_clustering.loc[df_clustering[REP] == rep]
    df_acc = df_acc.loc[df_acc[REP] == rep]

    df_merged = pd.merge(df_clustering, df_acc, on=[X1, X2, BUDGET, CLU_TIMEWINDOW])

    sb.set_theme()

    g = sb.relplot(data=df_merged, x=X1, y=X2, kind='scatter', palette="deep", edgecolor='White', linewidth=0.5, hue=GT, col=CLU_TIMEWINDOW, row=BUDGET)
    df_labeled = df_merged.loc[~np.isnan(df_merged['Label_x'])]
    #g.map(add_points, X1, X2, dataframe=df_labeled)
    axes = g.axes

    center_radi = df_clustering[[C1, C2, RADIUS, BUDGET, CLU_TIMEWINDOW]].drop_duplicates().to_numpy()
    # Sort the array by column Buget and then column Timewindow
    sorted_indices = np.lexsort((center_radi[:, 4], center_radi[:, 3]))
    center_radi = center_radi[sorted_indices]
    # Add Cluster Radi
    n_budget = len(np.unique(df_merged[BUDGET]))
    n_timewindow = len(np.unique(df_merged[CLU_TIMEWINDOW]))
    n_cluster = len(np.unique(df_merged['Cluster_x']))

    for b in range(n_budget):
        for t in range(n_timewindow):
            for c in range(n_cluster):
                c1 = center_radi[c + b* n_budget + t * n_timewindow][0]
                c2 = center_radi[c + b * n_budget + t * n_timewindow][1]
                radi = center_radi[c + b * n_budget + t * n_timewindow][2]
                axes[b][t].add_patch(patches.Circle([c1, c2], radius=radi * 3, fill=False, color="turquoise"))

    image_filepath = os.path.join(this_dir, "..", target_directory, 'output_clustering.pdf')
    save_image(image_filepath)

