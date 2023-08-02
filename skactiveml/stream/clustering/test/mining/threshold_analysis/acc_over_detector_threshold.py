import numpy as np
from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, CLU_TIMEWINDOW, TIMESTEP, N_CLUSTER, DETECTOR_THRESHOLD
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os

if __name__ == '__main__':
    smoothing_window = 300

    hue = CLASSIFIER
    col = N_CLUSTER
    row = BUDGET


    file_name = 'output_acc_over_threshold_per_' + hue + '_' + col + '_' + row + '.pdf'

    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "..", "..", target_directory, 'accuracy_time_window.csv')
    df = pd.read_csv(csv_filepath)

    df[CLU_TIMEWINDOW] = df[CLU_TIMEWINDOW].replace(np.inf, "Infinity")
    df[ACCURACY] = df[ACCURACY].rolling(window=smoothing_window).mean()

    sb.set_theme()

    f = sb.relplot(data=df, x=DETECTOR_THRESHOLD, y=ACCURACY, kind="line", hue=hue, palette='tab10')

    image_filepath = os.path.join(this_dir, "..", "..", target_directory, file_name)

    save_image(image_filepath)

