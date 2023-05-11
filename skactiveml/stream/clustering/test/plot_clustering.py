import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb

REP = "rep"
ROUND = "round"
APPROACH = "Approach"
CLUSTER = "Cluster"
TIME = "time"
BUDGET = "Budget"
X1 = "X1"
X2 = "X2"

df = pd.read_csv('target/test.csv')

#Plotting
fig = plt.figure()
ax = fig.add_subplot()

#Plot Data
sb.scatterplot(data=df, x=X1, y=X2, hue=CLUSTER, ax=ax)


ax.axis('equal')

plt.show()