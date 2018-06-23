import numpy as np
import test

arr = np.array([[1.0, 2.0], [-1.0, 3.5]])
test.tsne(arr, noDims=2, perplexity=20, theta=0.5)
