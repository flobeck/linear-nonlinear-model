from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState
import matplotlib.pyplot as plt

def plot_gallery(title, images, n_col, n_row):
    plt.figure(figsize=(2. * n_col, 2. * n_row))
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i+1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape((64, 64)),
                   cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(()); plt.yticks(())
    plt.tight_layout()
    plt.show()
