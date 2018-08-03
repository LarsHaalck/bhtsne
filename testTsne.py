#######################################
# FIlE SHOULD BE USED FROM PARENT DIR #
#######################################

import numpy as np
from tsne import tsne_run
import matplotlib.pyplot as plt

X = np.loadtxt("tsne/mnist2500_X.txt")
labels = np.loadtxt("tsne/mnist2500_labels.txt")
Y = tsne_run(X)

plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
plt.show()
