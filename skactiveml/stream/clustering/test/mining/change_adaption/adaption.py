import numpy as np
from matplotlib import pyplot as plt, patches

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, TIMESTEP, LABEL_FREQUENCY, CLUSTER, GT, Y, X1, X2, C1, C2, RADIUS, CLU_TIMEWINDOW
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os

if __name__ == '__main__':

    n_reps_to_plot = 3
    rep = 12

    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'

    csv_clustering_filepath = os.path.join(this_dir, "..", "..", target_directory, 'adaption.csv')
    df_adapt = pd.read_csv(csv_clustering_filepath)

    approaches = np.unique(df_adapt[CLASSIFIER])
    df_adapt = df_adapt.dropna(subset=[LABEL])
    df_adapt[LABEL] = df_adapt[LABEL].astype(int)

    csv_clustering_filepath = os.path.join(this_dir, "..", "..", target_directory, 'accuracy_time_window.csv')

    sb.set_theme()

    df_adapt = df_adapt[df_adapt[REP] == rep]

    standard_palette = sb.color_palette()

    color = [standard_palette[0], 'darkolivegreen', 'purple', 'black', standard_palette[1]]

    for approach in approaches:
        image_name = 'adaption' + approach + '.pdf'
        image_filepath = os.path.join(this_dir, "..", "..", target_directory, image_name)
        df_tmp = df_adapt[df_adapt[CLASSIFIER] == approach]
        g = sb.relplot(data=df_tmp, x=X1, y=X2, hue=LABEL, kind='scatter', palette=color)

        for ax in g.axes.flat:
            ax.set_xlabel(r'$x_1$', fontsize=16)  # Adjust x axis label font size
            ax.set_ylabel(r'$x_2$', fontsize=16)  # Adjust y axis label font size
            ax.tick_params(axis='both', labelsize=13)

        g._legend.set_bbox_to_anchor((0.95, 0.7))

        plt.tight_layout()

        leg = g.legend
        leg.set_title('Class')
        leg.get_title().set_fontsize('x-large')

        for t in leg.get_texts():
            t.set_fontsize(15)
        g.savefig(image_filepath)
    df_acc = pd.read_csv(csv_clustering_filepath)


    df_acc = df_acc[df_acc[REP] == rep]
    df_acc = df_acc[df_acc[CLASSIFIER] == 'ZliobaiteRefit']
    df_acc[LABEL] = df_acc[LABEL].astype(int)

    g = sb.relplot(data=df_acc, x=X1, y=X2, hue=LABEL, kind='scatter', palette=color)

    for ax in g.axes.flat:
        ax.set_xlabel(r'$x_1$', fontsize=16)  # Adjust x axis label font size
        ax.set_ylabel(r'$x_2$', fontsize=16)  # Adjust y axis label font size
        ax.tick_params(axis='both', labelsize=13)

    plt.tight_layout()

    leg = g.legend
    leg.set_title('Class')
    leg.get_title().set_fontsize('x-large')

    for t in leg.get_texts():
        t.set_fontsize(15)

    g._legend.set_bbox_to_anchor((0.95, 0.7))

    image_filepath = os.path.join(this_dir, "..", "..", target_directory, 'adaption_all.pdf')
    g.savefig(image_filepath)

