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
    col = DATASET

    csv_name = 'n_cluster_minimal.csv'
    file_name = 'n_cluster_analysis.pdf'

    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "..", "..", target_directory, csv_name)
    df = pd.read_csv(csv_filepath)

    image_filepath = os.path.join(this_dir, "..", "..", target_directory, file_name)

    #df = df.loc[df[DATASET] != 'Electricity']

    df = df.groupby([REP, BUDGET, CLASSIFIER, N_CLUSTER, DETECTOR_THRESHOLD, DATASET], as_index=False)[ACCURACY].mean()
    sb.set_theme()


    sb.set_theme()

    order_list = ['CORA-SP', 'CORA-SE', 'CORA-EP', 'CORA-EE']


    g = sb.relplot(data=df,
                   x=N_CLUSTER,
                   y=ACCURACY,
                   kind="line",
                   hue=hue,
                   col=col,
                   hue_order=order_list,
                   palette='tab10',
                   facet_kws={'sharey': True})

    g.set_titles(col_template="{col_name}")

    #save_image(image_filepath)
    sb.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False,
    )
    g.legend

    v_positions = {
        'Hyperplane': 2,
        'Electricity': 2,
        'RbfGenerator': 15
    }

    for ax in g.axes.flat:
        col_var = ax.get_title() # Extract column variable value from title
        x_pos = v_positions[col_var]
        ax.axvline(x=x_pos, color='darkmagenta', linestyle='--')

    # Save the figure as a PDF file
    g.fig.savefig(image_filepath, bbox_inches='tight')

    plt.close()  # Closes the
