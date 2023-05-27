import numpy as np
from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os


if __name__ == '__main__':
    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "..", target_directory, 'output.csv')
    df = pd.read_csv(csv_filepath)
    df_budget = df.groupby([BANDWIDTH, CLASSIFIER, REP, BUDGET])[ACCURACY].mean().reset_index()

    sb.set_theme()

    f = sb.relplot(data=df_budget, x=BUDGET, y=ACCURACY, kind="line", hue=CLASSIFIER)
    f.set(title='Accuracy')

    # Plotting label aquisition counts (Probably not needed anymore)
    '''
    label_acquisition = df.groupby([BANDWIDTH, CLASSIFIER, REP, BUDGET, CLU_TIMEWINDOW])[LABEL].apply(lambda x: (pd.notna(x)).sum())
    df_label_acquisition = pd.DataFrame(label_acquisition.reset_index())

    g = sb.relplot(
        data=df_label_acquisition, x=BUDGET, y=LABEL, kind="line", hue=CLASSIFIER
    )
    g.set(title='Label Acquisition Count')
    '''

    image_filepath = os.path.join(this_dir, "..", target_directory, 'output_accuracy.pdf')

    save_image(image_filepath)

