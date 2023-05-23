from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, TIMESTEP, LABEL_FREQUENCY
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os


if __name__ == '__main__':
    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "..", target_directory, 'output.csv')
    df = pd.read_csv(csv_filepath)

    sb.set_theme()

    f = sb.relplot(data=df, x=TIMESTEP, y=LABEL_FREQUENCY, kind="line", hue=CLASSIFIER, col=CLASSIFIER, row=BUDGET)
    #f.set(title='Label Frequency Predictions')

    image_filepath = os.path.join(this_dir, "..", target_directory, 'output_lbl_frequency.pdf')

    save_image(image_filepath)

