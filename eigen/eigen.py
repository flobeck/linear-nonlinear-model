import numpy as np
from gallery import *
from waveformaudio import *

def eigen(p):
    from numpy import linalg

    p = np.matrix(p)
    p_ = p.mean(axis=0)
    P = p - p_

    evalues,evectors = linalg.eig(np.dot(P, np.transpose(P)))

    args  = evalues.argsort()
    evectors = evectors[args]
    evectors = evectors[::-1]

    P_evectors = [(np.dot(np.transpose(P), np.transpose(evectors[i])))
                  for i in range(0, len(evalues))]
    eigen = map(lambda e: e/np.linalg.norm(e), P_evectors)

    return eigen


###
# 'Eigenface':
###

from sklearn.datasets import fetch_olivetti_faces

n_row, n_col = 10, 10; n_eigenfaces = n_row * n_col
faces = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0)).data
eigenfaces = eigen(faces)

plot_gallery("Eigenfaces", eigenfaces[:n_eigenfaces], n_col, n_row)



###
# 'Eigensound':
###
def load(directory):
    if not os.path.isfile(directory + '.npy'):
        data = getWAV('./' + directory.title())
        np.save(directory, data)
        return data
    else:
        return np.load(directory + '.npy')

#klavier = load('Klavier')

#eigenklavier = eigen(klavier)
#normalize_write(eigenklavier, 'k')
