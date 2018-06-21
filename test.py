import numpy as np
import bhtsne
import matplotlib.pyplot as plt

data = np.loadtxt("mnist2500_X.txt", skiprows=1)

embedding_array = bhtsne.run_bh_tsne(data, initial_dims=data.shape[1])

print(embedding_array.shape)
