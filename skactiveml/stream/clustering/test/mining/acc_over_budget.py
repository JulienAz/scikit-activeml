import os

import pandas as pd
import seaborn as sb

from skactiveml.stream.clustering.test.ExperimentLogger.clu_stream_performance_logger import DATASET, CLASSIFIER, REP, BUDGET, ACCURACY

hue = CLASSIFIER
col = DATASET

file_name = 'minimal_acc.pdf'

this_dir = os.path.split(__file__)[0]
target_directory = 'target'
csv_filepath = os.path.join(this_dir, "..", target_directory, 'minimal.csv')
df = pd.read_csv(csv_filepath)

df = df.groupby([REP, BUDGET, CLASSIFIER, DATASET], as_index=False)[ACCURACY].mean()
sb.set_theme()

# Artificial
datasets_to_include = ['Hyperplane', 'ChessBoard', 'RbfGenerator', 'SEA']

#Realworld
#datasets_to_include = ['Electricity', 'Airlines', 'Covertype', 'Pokerhand']

df = df[df[DATASET].isin(datasets_to_include)]


#g = sb.relplot(data=df, x=BUDGET, y=ACCURACY, kind="line", hue=hue, col=col, col_wrap=3, palette='tab10', facet_kws={'sharey': False})

# Define your hue order
hue_order = ['CORA-SP', 'CORA-SE', 'CORA-EP', 'CORA-EE', 'Zliobaite', 'PEFAL', 'OPAL-NA']


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
               hue_order=hue_order,
               palette=palette_dict,  # Use the palette dictionary here
               facet_kws={'sharey': False})

g.set_titles(col_template="{col_name}")

sb.move_legend(
    g, "lower center",
    bbox_to_anchor=(.5, 1), ncol=len(hue_order), title=None, frameon=False,
)

g.legend

image_filepath = os.path.join(this_dir, "..", target_directory, file_name)

g.savefig(image_filepath)