import numpy as np
from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, CLU_TIMEWINDOW, TIMESTEP, N_CLUSTER, DETECTOR_THRESHOLD, DATASET
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os

if __name__ == '__main__':
    hue = CLASSIFIER
    col = N_CLUSTER
    row = DATASET

    csv_name = 'threshold_minimal.csv'
    file_name = 'threshold_analysis.pdf'

    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "..", "..", target_directory, csv_name)
    df = pd.read_csv(csv_filepath)

    image_filepath = os.path.join(this_dir, "..", "..", target_directory, file_name)

    df = df.loc[df[DATASET] != 'Electricity']

    df = df.groupby([REP, BUDGET, CLASSIFIER, N_CLUSTER, DETECTOR_THRESHOLD, DATASET], as_index=False)[ACCURACY].mean()
    sb.set_theme()


    sb.set_theme()

    order_list = ['CORA-SP', 'CORA-SE', 'CORA-EP', 'CORA-EE']
    cora_colors = ['darkolivegreen', 'firebrick', 'darkcyan', 'purple']

    g = sb.relplot(data=df,
                   x=DETECTOR_THRESHOLD,
                   y=ACCURACY,
                   kind="line",
                   hue=hue,
                   col=col,
                   row=row,
                   hue_order=order_list,
                   palette=cora_colors,
                   facet_kws={'sharey': 'row'})

    g.set_titles(col_template="{col_name} Cluster", row_template="{row_name}")

    sb.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False,
    )

    for ax in g.axes.flat:
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)  # Adjust x axis label font size
        ax.set_ylabel(ax.get_ylabel(), fontsize=15)  # Adjust y axis label font size
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(ax.get_title(), fontsize=14)

    leg = g.legend

    for t in leg.get_texts():
        t.set_fontsize(13)

    # Save the figure as a PDF file
    g.tight_layout()
    g.savefig(image_filepath)
    #g.fig.savefig(image_filepath, bbox_inches='tight')

    plt.close()  # Closes the
