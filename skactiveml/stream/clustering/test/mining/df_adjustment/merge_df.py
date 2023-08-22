
import pandas as pd
import seaborn as sb
import os


if __name__ == '__main__':
    this_dir = os.path.split(__file__)[0]
    target_directory = os.path.join(this_dir, "..", "..", 'target')

    csv_1_filepath = os.path.join(target_directory, 'minimal.csv')

    csv_2_filepath = os.path.join(target_directory, 'accuracy_time_window.csv')

    #csv_3_filepath = os.path.join(target_directory, 'n_cluster_rbf.csv')

    df1 = pd.read_csv(csv_1_filepath)
    df2 = pd.read_csv(csv_2_filepath)
    #df3 = pd.read_csv(csv_3_filepath)

    concatenated_df = pd.concat([df1, df2], ignore_index=True)

    target_filepath = os.path.join(target_directory, 'minimal.csv')

    concatenated_df.to_csv(target_filepath, index=False)

