import numpy as np
from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, TIMESTEP, LABEL_FREQUENCY, CLUSTER, GT, Y
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os


if __name__ == '__main__':
    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "..", target_directory, 'output.csv')
    df = pd.read_csv(csv_filepath)
    df_clustering = df.loc[df[CLASSIFIER] == 'ClusteringBased']
    n_cluster = np.unique(df_clustering[CLUSTER])

    sb.set_theme()

    f = sb.displot(data=df_clustering, x=CLUSTER, hue=GT, palette="tab10", multiple="stack", bins=n_cluster)
    f.set(title='GT_Label Distritubtion')

    df_clustering_al = df_clustering[df_clustering[LABEL].notna()]
    g = sb.displot(data=df_clustering, x=CLUSTER, hue=LABEL, palette="tab10", multiple="stack", bins=n_cluster)
    g.set(title='AL_Label Distritubtion')

    image_filepath = os.path.join(this_dir, "..", target_directory, 'output_cluster_dist.pdf')
    save_image(image_filepath)

