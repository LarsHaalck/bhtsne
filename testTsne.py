import numpy as np
from bhtsne_cpp11 import tsne_run
import matplotlib.pyplot as plt

X = np.loadtxt("bhtsne_cpp11/mnist2500_X.txt")
labels = np.loadtxt("bhtsne_cpp11/mnist2500_labels.txt")
Y = tsne_run(X)

plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
plt.show()
