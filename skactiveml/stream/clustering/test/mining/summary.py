import os

import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import DATASET, CLASSIFIER, REP, BUDGET, ACCURACY

hue = CLASSIFIER
col = DATASET

file_name = 'merged_summary.csv'

this_dir = os.path.split(__file__)[0]
target_directory = 'target'
csv_filepath = os.path.join(this_dir, "..", target_directory, 'minimal.csv')
df = pd.read_csv(csv_filepath)

df = df.groupby([DATASET, CLASSIFIER], as_index=False)[ACCURACY].mean()

df_pivot = df.pivot(index=DATASET, columns=CLASSIFIER, values=ACCURACY)
df_pivot = df_pivot.reset_index()

csv_filepath = os.path.join(this_dir, "..", target_directory, file_name)
df_pivot.to_csv(csv_filepath, index=False)