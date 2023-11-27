import pandas as pd
import seaborn as sb
import os


if __name__ == '__main__':
    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'


    csv_1_filepath = os.path.join(this_dir, "..", target_directory, 'threshold_minimal.csv')

    df = pd.read_csv(csv_1_filepath)

    df['n_Cluster'] = df['n_Cluster'].replace(23, 25)

    target_filepath = os.path.join(this_dir, "..", target_directory, 'threshold_merged.csv')

    df.to_csv(target_filepath, index=False)
