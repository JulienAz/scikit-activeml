import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save_image(filename):
    # PdfPages is a wrapper around pdf
    # file so there is no clash and
    # create files with no error.
    p = PdfPages(filename)

    # get_fignums Return list of existing
    # figure numbers
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]

    # iterating over the numbers in list
    for fig in figs:
        # and saving the files
        fig.savefig(p, format='pdf')

        # close the object
    p.close()


def map_edgecolor(val):
    if np.isnan(val):
        return "white"
    else:
        return "red"


def map_linewidth(val):
    if np.isnan(val):
        return 0.5
    else:
        return 1


def get_center_radi(df):
    test = 0


# Columns of Dataframe

REP = "rep"
ROUND = "round"
APPROACH = "Approach"
CLUSTER = "Cluster"
TIME = "time"
BUDGET = "Budget"
X1 = "X1"
X2 = "X2"
Y = "Label"
GT = "GT_Label"

# Begin of Plot Script

file_name = 'iris/iris_10000_noise_0,1_cluster_10_OPAL.csv'

df = pd.read_csv(file_name)

# Plotting
sb.set_theme()

# Plotting Clusters
fig1, axis = plt.subplots(1, 1)
edgecolors = df[Y].apply(map_edgecolor)
edge_width = df[Y].apply(map_linewidth)

df_unlabeled = df.loc[np.isnan(df[Y])]

# sb.scatterplot(data=df, x=X1, y=X2, palette="deep", hue=GT, ax=axis)
sb.scatterplot(data=df.loc[np.isnan(df[Y])], x=X1, y=X2, palette="deep", hue=GT, ax=axis, edgecolor='White',
               linewidth=0.5)
sb.scatterplot(data=df.loc[~np.isnan(df[Y])], x=X1, y=X2, palette="deep", hue=GT, ax=axis, edgecolor='red', linewidth=1,
               zorder=10, legend=False)
# Add Cluster Radi
# for i, c in enumerate(center):
#    axis.add_patch(patches.Circle(c, radius=radi[i] * cluStream.r_factor, fill=False, color="turquoise"))

save_image('target/tmp.pdf')
