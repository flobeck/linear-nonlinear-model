import numpy as np

dt     = 0.1

def poisson_process(t, fvalue):
    if fvalue*dt > np.random.uniform(0,1):
        return 1
    else:
        return 0

def spiking(fvalue):
    spikes = []
    for i in range(0,250):
        spikes.append(poisson_process(i,fvalue[i]))
        spikes.append(0)
    return spikes
