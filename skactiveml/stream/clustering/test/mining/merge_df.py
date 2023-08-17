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


    csv_1_filepath = os.path.join(this_dir, "..", target_directory, 'accuracy_time_window.csv')

    csv_2_filepath = os.path.join(this_dir, "..", target_directory, 'accuracy_budget_elec_hyper.csv')

    df1 = pd.read_csv(csv_1_filepath)
    df2 = pd.read_csv(csv_2_filepath)

    concatenated_df = pd.concat([df1, df2], ignore_index=True)

    target_filepath = os.path.join(this_dir, "..", target_directory, 'acc_over_budget_merged.csv')

    concatenated_df.to_csv(target_filepath, index=False)

