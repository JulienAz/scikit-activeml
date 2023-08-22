import numpy as np
from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, CLU_TIMEWINDOW, TIMESTEP, N_CLUSTER, DETECTOR_THRESHOLD, DATASET, \
    CHANGE_DETECTION
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os

if __name__ == '__main__':
    hue = CLASSIFIER
    col = DATASET

    csv_name = 'change_detection.csv'
    file_name = 'change_detection.pdf'

    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "..", "..", target_directory, csv_name)
    df = pd.read_csv(csv_filepath)

    image_filepath = os.path.join(this_dir, "..", "..", target_directory, file_name)

    #df = df.loc[df[DATASET] != 'Electricity']

    df = df.groupby([REP, BUDGET, CLASSIFIER, DATASET], as_index=False)[CHANGE_DETECTION].sum()


    order_list = ['CORA-SP', 'CORA-SE', 'CORA-EP', 'CORA-EE']

    sb.set_theme()

    g = sb.barplot(
        df,
        x=DATASET,
        y=CHANGE_DETECTION,
        hue=CLASSIFIER
    )
    #g.set_titles(col_template="{col_name}")

    #save_image(image_filepath)
    #sb.move_legend(
    #    g, "lower center",
    #    bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False,
    #)
    #g.legend

    v_positions = {
        'Hyperplane': 2,
        'Electricity': 2,
        'RbfGenerator': 15
    }

    #for ax in g.axes.flat:
    #    col_var = ax.get_title() # Extract column variable value from title
    #    x_pos = v_positions[col_var]
    #    ax.axvline(x=x_pos, color='darkmagenta', linestyle='--')#

    # Save the figure as a PDF file
    g.get_figure().savefig(image_filepath, bbox_inches='tight')

    plt.close()  # Closes the
