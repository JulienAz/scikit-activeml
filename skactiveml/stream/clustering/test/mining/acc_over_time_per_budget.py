import numpy as np
from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, CLU_TIMEWINDOW, TIMESTEP
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os


if __name__ == '__main__':
    # Window for smoothening the accuracy
    smoothing_window = 300

    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "..", target_directory, 'output.csv')
    df = pd.read_csv(csv_filepath)

    sb.set_theme()

    df[ACCURACY] = df[ACCURACY].rolling(window=smoothing_window).mean()
    f = sb.relplot(data=df, x=TIMESTEP, y=ACCURACY, kind="line", hue=CLASSIFIER, col=BUDGET, col_wrap=3, errorbar=None)

    # Plotting label aquisition counts (Probably not needed anymore)
    '''label_acquisition = df.groupby([BANDWIDTH, CLASSIFIER, REP, BUDGET])[LABEL].apply(lambda x: (pd.notna(x)).sum())
    df_label_acquisition = pd.DataFrame(label_acquisition.reset_index())

    g = sb.relplot(
        data=df_label_acquisition, x=BUDGET, y=LABEL, kind="line", hue=CLASSIFIER
    )
    g.set(title='Label Acquisition Count')'''

    image_filepath = os.path.join(this_dir, "..", target_directory, 'output_acc_over_time.pdf')

    save_image(image_filepath)

