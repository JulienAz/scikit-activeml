import numpy as np
from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import ACCURACY, BUDGET, \
    CLASSIFIER, BANDWIDTH, REP, LABEL, CLU_TIMEWINDOW, TIMESTEP, N_CLUSTER, DETECTOR_THRESHOLD, DATASET
from skactiveml.stream.clustering.util import save_image, run_async

import pandas as pd
import seaborn as sb
import os

if __name__ == '__main__':

    this_dir = os.path.split(__file__)[0]

    target_directory = os.path.join(this_dir, "..", "..", 'target')
    filename = 'merged.csv'
    filename2= 'merged_new'

    csv_filepath = os.path.join(this_dir, "..", target_directory, filename)

    df = pd.read_csv(csv_filepath)

    df = df[df[CLASSIFIER] != 'PEFAL']

    csv_filepath = os.path.join(target_directory, filename2)
    df.to_csv(csv_filepath)



