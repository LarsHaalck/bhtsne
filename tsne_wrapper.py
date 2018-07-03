import numpy as np
import sys
import tsne_module

def tsne_run(X, targetDims = 2, pcaDims = 50, perplexity = 30.0, theta = 0.5,
        learningRate = 200.0, maxIt = 1000, distMeasure = 'euclidean', numThreads = 0):

    print("Using module file: " + tsne_module.__file__)
    print("Preprocessing the data using PCA...")
    if pcaDims > 0:
        (n, d) = X.shape
        X = X - np.tile(np.mean(X, 0), (n, 1))
        (l, M) = np.linalg.eig(np.dot(X.T, X))
        X = np.dot(X, M[:, 0:pcaDims]).real

    X = np.ascontiguousarray(X)
    return tsne_module.run(X, no_dims=targetDims, perplexity=perplexity, theta=theta,
            learning_rate=learningRate, max_iter=maxIt, dist_measure = distMeasure,
            num_threads = numThreads)
