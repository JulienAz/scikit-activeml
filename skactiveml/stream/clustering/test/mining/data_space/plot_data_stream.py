# Generating Datastream
import os

import pandas as pd
from matplotlib import pyplot as plt

from skactiveml.stream.clustering.data import generate_data
from skactiveml.stream.clustering.data.datasets import CHESSBOARD, HYPERPLANE
import seaborn as sb

dataset = CHESSBOARD
len = 4000
random_state = 14

n_features=2
mag_change=0.2

this_dir = os.path.split(__file__)[0]
target_directory = os.path.join(this_dir, "..", "..", 'target')
filename = dataset['name'] + '.pdf'

image_filepath = os.path.join(this_dir, "..", target_directory, filename)

palette = sb.color_palette('magma', 8)
palette = [palette[2], palette[6]]

X, y = generate_data(dataset, len, 0, random_state=random_state, stream_length=len,
                     start_point=0, n_features=n_features, mag_change=mag_change)

df = pd.DataFrame(X, columns=['x', 'y'])
df['Class'] = y

# Plot using DataFrame
g = sb.scatterplot(data=df, x='x', y='y', hue='Class', palette=palette)

g.set_xlabel(r'$x_1$', fontsize=14)
g.set_ylabel(r'$x_2$', fontsize=14)

# Increase the font size of the legend
legend = g.legend(fontsize=12, loc='lower right')
legend.set_title("Class", prop={'size': 14})
plt.tight_layout()
#g.figure.set_size_inches(4, 4)
g.figure.savefig(image_filepath)