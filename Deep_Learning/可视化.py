# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 17:11:48 2021

@author: TAY
"""

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
# digits.data        类型：Array of float64  大小:(1797,64)
X_tsne = TSNE(n_components=2, random_state=33).fit_transform(digits.data)
X_pca = PCA(n_components=2).fit_transform(digits.data)

font = {"color": "darkred",
        "size": 13, 
        "family" : "serif"}

plt.style.use("dark_background")
plt.figure(figsize=(8.5, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target, alpha=0.6, 
            cmap=plt.cm.get_cmap('rainbow', 10))

plt.title("t-SNE", fontdict=font)
cbar = plt.colorbar(ticks=range(10)) 
cbar.set_label(label='digit value', fontdict=font)
plt.clim(-0.5, 9.5)
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, alpha=0.6, 
            cmap=plt.cm.get_cmap('rainbow', 10))

plt.title("PCA", fontdict=font)
cbar = plt.colorbar(ticks=range(10))
cbar.set_label(label='digit value', fontdict=font)
plt.clim(-0.5, 9.5)
plt.tight_layout()


# DESCR：str 1
# data：Array of float64 (1797, 64)
# images：Array of float64 (1797, 8, 8)
# target：Array of int32 (1797,)
# target_names：Array of int32 (10,)