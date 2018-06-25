import numpy as np
import tsne

arr = np.random.rand(1000, 500)
res = tsne_omp.run(arr, noDims=2, perplexity=20, theta=0.5)
print(res.shape)
