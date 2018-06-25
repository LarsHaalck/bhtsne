import numpy as np
import sys

def tsne(X, targetDims = 2, pcaDims = 50, perplexity = 30.0, theta = 0.5,
        learningRate = 200.0, maxIt = 1000, numThreads = 0):

    print("Preprocessing the data using PCA...")
    if pcaDims > 0:
        (n, d) = X.shape
        X = X - np.tile(np.mean(X, 0), (n, 1))
        (l, M) = np.linalg.eig(np.dot(X.T, X))
        X = np.dot(X, M[:, 0:pcaDims]).real

    X = np.ascontiguousarray(X)
    module = None
    if 'tsne_python' in sys.modules:
        module = sys.modules['tsne_python']
    if 'tsne_cpp' in sys.modules:
        module = sys.modules['tsne_cpp']
    return module.tsne(X, no_dims=targetDims, perplexity=perplexity, theta=theta,
            learning_rate=learningRate, max_iter=maxIt, num_threads = numThreads)
