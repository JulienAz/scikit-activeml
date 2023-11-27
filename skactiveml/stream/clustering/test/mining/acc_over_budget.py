import os

import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import DATASET, CLASSIFIER, REP, BUDGET, ACCURACY

hue = CLASSIFIER
col = DATASET

file_name = 'minimal_acc.pdf'

this_dir = os.path.split(__file__)[0]
target_directory = 'target'
csv_filepath = os.path.join(this_dir, "..", target_directory, 'merged.csv')
df = pd.read_csv(csv_filepath)

df = df.groupby([REP, BUDGET, CLASSIFIER, DATASET], as_index=False)[ACCURACY].mean()
sb.set_theme()

# Artificial
#datasets_to_include = ['Hyperplane', 'ChessBoard', 'RbfGenerator', 'SEA']

#Realworld
datasets_to_include = ['Electricity', 'Airlines', 'Covertype', 'PokerHand']

df = df[df[DATASET].isin(datasets_to_include)]


#g = sb.relplot(data=df, x=BUDGET, y=ACCURACY, kind="line", hue=hue, col=col, col_wrap=3, palette='tab10', facet_kws={'sharey': False})

# Define your hue order
hue_order = ['CORA-SP', 'CORA-SE', 'CORA-EP', 'CORA-EE', 'Zliobaite', 'PEFAL', 'OPAL-NA']
col_order = ['Electricity', 'Airlines', 'Covertype', 'PokerHand']

# Get the default seaborn colors and 'rocket' palette colors
default_colors = sb.color_palette()  # by default gives the deep palette
rocket_colors = sb.color_palette("rocket")

# Decide how many colors you need from each
num_cora = sum(1 for hue in hue_order if "CORA" in hue)
num_others = len(hue_order) - num_cora

# Subset the colors
cora_colors = default_colors[:num_cora]
other_colors = default_colors[:3]

cora_colors = ['darkolivegreen', 'firebrick','darkcyan', 'purple']

# Combine colors based on your criteria
combined_colors = cora_colors + other_colors

# Create a dictionary for the palette
palette_dict = dict(zip(hue_order, combined_colors))

# Now use this palette_dict in your relplot
g = sb.relplot(data=df,
               x=BUDGET,
               y=ACCURACY,
               kind="line",
               hue=hue,
               col=col,
               col_wrap=2,
               col_order=col_order,
               hue_order=hue_order,
               palette=palette_dict,  # Use the palette dictionary here
               height=4,         # Adjust the height of each facet
               aspect=1,
               facet_kws={'sharey': False})

g.set_titles(col_template="{col_name}", size=12)

sb.move_legend(
    g, "lower center",
    bbox_to_anchor=(.485, 0.99), ncol=len(hue_order), title=None, frameon=False,
)
leg = g.legend

for t in leg.get_texts():
    t.set_fontsize(10)

#for ax in g.axes.flat:
#    ax.set_xlabel(ax.get_xlabel(), fontsize=14)  # Adjust x axis label font size
#    ax.set_ylabel(ax.get_ylabel(), fontsize=14)  # Adjust y axis label font size
#    ax.tick_params(axis='both', labelsize=12)


#g.set_axis_labels(BUDGET, ACCURACY, fontsize=14)

g.legend
plt.tight_layout()
#plt.subplots_adjust(left=0.3, right=0.8, bottom=0.2, top=0.9)
image_filepath = os.path.join(this_dir, "..", target_directory, file_name)

g.savefig(image_filepath)