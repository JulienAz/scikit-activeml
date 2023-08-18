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

    file_name = 'acc_over_budget_per_classifier.pdf'

    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "..", target_directory, 'acc_over_budget_merged.csv')
    df = pd.read_csv(csv_filepath)

    df = df.groupby([REP, BUDGET, CLASSIFIER, DATASET], as_index=False)[ACCURACY].mean()
    sb.set_theme()

    f = sb.relplot(data=df, x=BUDGET, y=ACCURACY, kind="line", hue=hue, col=col, palette='tab10', facet_kws={'sharey': False})

    image_filepath = os.path.join(this_dir, "..", target_directory, file_name)

    save_image(image_filepath)

