import numpy as np
from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, CLU_TIMEWINDOW
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os


if __name__ == '__main__':
    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "..", target_directory, 'accuracy_time_window.csv')
    df = pd.read_csv(csv_filepath)
    df_budget = df.groupby([BANDWIDTH, CLASSIFIER, REP, BUDGET, CLU_TIMEWINDOW])[ACCURACY].mean().reset_index()

    sb.set_theme()

    df_budget[CLU_TIMEWINDOW] = df_budget[CLU_TIMEWINDOW].replace(np.inf, "Infinity")
    f = sb.relplot(data=df_budget, x=BUDGET, y=ACCURACY, kind="line", hue=CLU_TIMEWINDOW)
    f.set(title='Accuracy')

    image_filepath = os.path.join(this_dir, "..", target_directory, 'output_acc_over_budget_timewindow.pdf')

    save_image(image_filepath)

