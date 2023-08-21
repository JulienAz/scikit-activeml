
import pandas as pd
import seaborn as sb
import os


if __name__ == '__main__':
    this_dir = os.path.split(__file__)[0]
    target_directory = 'target'


    csv_1_filepath = os.path.join(this_dir, "..", target_directory, 'n_cluster_elec.csv')

    csv_2_filepath = os.path.join(this_dir, "..", target_directory, 'n_cluster_hyperplane.csv')

    csv_3_filepath = os.path.join(this_dir, "..", target_directory, 'n_cluster_rbf.csv')

    df1 = pd.read_csv(csv_1_filepath)
    df2 = pd.read_csv(csv_2_filepath)
    df3 = pd.read_csv(csv_3_filepath)

    concatenated_df = pd.concat([df1, df2, df3], ignore_index=True)

    target_filepath = os.path.join(this_dir, "..", target_directory, 'n_cluster_merged.csv')

    concatenated_df.to_csv(target_filepath, index=False)

