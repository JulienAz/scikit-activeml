import np as np
import numpy as np
from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, TIMESTEP, LABEL_FREQUENCY, CLUSTER, GT, Y
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os


if __name__ == '__main__':

    # Number of Reps the plot should be made of
    n_reps_to_plot = 3

    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "..", target_directory, 'output.csv')
    df = pd.read_csv(csv_filepath)

    # Taking only results of Clustering based approaches
    df_clu = df.loc[df[CLASSIFIER] == 'ClusteringBased']
    n_cluster = np.unique(df_clu[CLUSTER])

    # Randomly pick the reps to plot
    reps = np.unique(df_clu[REP])
    assert n_reps_to_plot <= len(reps)
    'Configured number of reps cannot be greater then reps in experiment csv'

    reps = np.random.choice(reps, size=n_reps_to_plot, replace=False)
    df_clu = df_clu.loc[df_clu[REP].isin(reps)]

    budget = np.random.choice(np.unique(df_clu[BUDGET]), size=1, replace=False)[0]

    sb.set_theme()

    # Plot cluster ground truth label distribution for random budget (is identical for all budgets)
    f = sb.displot(data=df_clu.loc[df_clu[BUDGET] == budget], x=CLUSTER, hue=GT, col=REP, col_wrap=3, palette="tab10", multiple="stack", bins=n_cluster)

    image_filepath = os.path.join(this_dir, "..", target_directory, 'output_cluster_sample_dist.pdf')
    save_image(image_filepath)

