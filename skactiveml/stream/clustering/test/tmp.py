import numpy as np

testArray = np.random.uniform(0, 1, (5, 100))

ls = np.sum(testArray, 0)
ss = np.sum(np.square(testArray), 0),