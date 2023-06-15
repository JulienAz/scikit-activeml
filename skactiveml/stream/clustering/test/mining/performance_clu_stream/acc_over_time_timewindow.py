import numpy as np
from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, CLU_TIMEWINDOW, TIMESTEP
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os


if __name__ == '__main__':
    smoothing_window = 300

    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "../..", target_directory, 'accuracy_time_window.csv')
    df = pd.read_csv(csv_filepath)

    sb.set_theme()

    df[CLU_TIMEWINDOW] = df[CLU_TIMEWINDOW].replace(np.inf, "Infinity")
    df[ACCURACY] = df[ACCURACY].rolling(window=smoothing_window).mean()
    f = sb.relplot(data=df, x=TIMESTEP, y=ACCURACY, kind="line", hue=CLU_TIMEWINDOW, col=BUDGET, col_wrap=3, errorbar=None)
    #f.set(title='Accuracy')

    image_filepath = os.path.join(this_dir, "../..", target_directory, 'output_acc_over_time_timewindow.pdf')

    save_image(image_filepath)

