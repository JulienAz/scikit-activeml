from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP
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
    plt.title('Accuracy')

    # Plotting label aquisition counts
    label_acquisition = df.groupby([BANDWIDTH, CLASSIFIER, REP, BUDGET])[Y].apply(lambda x: (pd.notna(x)).sum())
    df_label_acquisition = pd.DataFrame(label_acquisition.reset_index())
    plt.title('Acquisition Count')
    g = sb.relplot(
        data=df_label_acquisition, x=BUDGET, y=Y, kind="line", hue=CLASSIFIER
    )

    image_filepath = os.path.join(this_dir, "..", target_directory, 'output.pdf')

    save_image(image_filepath)

