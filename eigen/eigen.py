import numpy as np
from gallery import *
from waveformaudio import *

def eigen(p):
    from numpy import linalg

    p = np.matrix(p)
    p_ = p.mean(axis=0)
    P = p - p_

    L = np.dot(P, np.transpose(P))

    evalues,evectors = linalg.eig(L)

    args  = evalues.argsort()
    evectors = evectors[args]
    evectors = evectors[::-1]

    e = []

    for i in range(0, int(len(evalues))):
        Me  = np.dot(np.transpose(np.array(P)),
                     np.transpose(evectors[i]))

        normalize = Me/np.linalg.norm(Me)
        e.append(normalize)

    return e



###
# 'eigenface':
###

from sklearn.datasets import fetch_olivetti_faces

n_row, n_col = 10, 10; n_eigenfaces = n_row * n_col
faces = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0)).data
eigenface = eigen(faces)

plot_gallery("Eigenfaces", eigenface[:n_eigenfaces], n_col, n_row)


###
# 'eigensound':
###

#eigensound = eigen(getWAV('./test'))  #
#normalize_write(eigensound, 'X')
