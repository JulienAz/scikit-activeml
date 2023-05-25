import numpy as np
from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, TIMESTEP, LABEL_FREQUENCY
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os


if __name__ == '__main__':
    # Only plotting frequency prediction for Clustering based approach
    only_clust = True

    # Plotting randomly chosen reps, set None if average of all reps should be plotted
    n_reps_to_plot = 6

    budget_to_plot = 0.51

    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'
    csv_filepath = os.path.join(this_dir, "..", target_directory, 'output.csv')
    df = pd.read_csv(csv_filepath)

    df_frequencies = df[df[LABEL_FREQUENCY].notna()]

    if only_clust:
        df_frequencies = df_frequencies.loc[df_frequencies[CLASSIFIER] == 'ClusteringBased']

    sb.set_theme()

    if n_reps_to_plot is not None:
        reps = np.unique(df[REP])
        assert n_reps_to_plot <= len(reps)
        'Configured number of reps cannot be greater then reps in experiment csv'

        reps = np.random.choice(reps, size=n_reps_to_plot, replace=False)
        df_frequencies = df_frequencies.loc[df_frequencies[REP].isin(reps)]

        # Plotting for specific budget
        df_frequencies = df_frequencies.loc[df_frequencies[BUDGET] == budget_to_plot]

        #Plot the Frequency prediction with respect to Repetion
        f = sb.relplot(data=df_frequencies, x=TIMESTEP, hue=CLASSIFIER, y=LABEL_FREQUENCY, kind="line", col=REP,
                       col_wrap=3, errorbar=None)
    else:
        f = sb.relplot(data=df_frequencies, x=TIMESTEP, hue=CLASSIFIER, y=LABEL_FREQUENCY, kind="line", col=BUDGET,
                       col_wrap=3, errorbar=None)

    image_filepath = os.path.join(this_dir, "..", target_directory, 'output_lbl_frequency.pdf')

    save_image(image_filepath)

