import numpy as np
import sys
import tsne_module
import sklearn.decomposition as skd


def tsne_run(
        X,
        targetDims=2,
        pcaDims=50,
        perplexity=30.0,
        theta=0.5,
        learningRate=200.0,
        maxIt=1000,
        distMeasure='euclidean',
        numThreads=0):

    print("Using module file: " + tsne_module.__file__)
    if pcaDims > 0:
        print("Preprocessing the data using PCA...")
        oldDims = X.shape[1]
        pca = skd.PCA(n_components=pcaDims).fit(X)
        X = pca.transform(X)
        print("Explained variance is: " +
              str(pca.explained_variance_ratio_.sum()) +
              " in " +
              str(X.shape[1]) +
              " of " +
              str(oldDims) +
              " dimensions")
    else:
        print("Using all " + str(X.shape[1]) + " dimensions")

    X = np.ascontiguousarray(X)
    print("Performing T-SNE...")
    return tsne_module.run(
        X,
        no_dims=targetDims,
        perplexity=perplexity,
        theta=theta,
        learning_rate=learningRate,
        max_iter=maxIt,
        dist_measure=distMeasure,
        num_threads=numThreads)
