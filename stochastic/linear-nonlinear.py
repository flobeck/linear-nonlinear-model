import numpy as np
import pylab as plt
import math
import matplotlib.animation as anim
import poisson

def ricker(x,o):
    return 1* 2/((3.0*o)**(0.5)*(3.1415)**0.25) * \
        (1-x**2/o**2)*2.71**(-x**2/(2.0*o**2))

def gauss(x): return 2.4**(-(x-20)**2/(100.0))-0.2

def sigmoid(x): return 1.0/(1.0+math.exp(-x))

def f(x): return math.exp(x*0.5)


y = []; g = []
for i in np.linspace(0,5,250):
    y.append(np.random.random_sample()-np.random.random_sample())
    g.append(ricker((i-2.5),0.05))
#   g.append(sigmoid(i-2.5))


conv = np.convolve(y,g,'same')
x = np.linspace(0,5,250)
nonlinear = [f(c) for c in conv]


fig, ax = plt.subplots(5,1,facecolor='w', edgecolor='g')
fig.subplots_adjust(hspace = .5)
ax = ax.ravel()
repeat_length = 250


for i in range(5):
    ax[i] = plt.subplot(5, 1, i+1)
    ax[i].set_xlim([0,5])
    if i == 0:
        plt.plot(x, y, 'b-'); plt.title('stimulus')
    elif i == 1:
        plt.plot(x,g,'r-'); plt.title('kernel')
    elif i == 2:
        plt.plot(x,conv,'g-'); plt.title('stimulus * kernel')
    elif i == 3:
        plt.plot(x,nonlinear,'c-'); plt.title('nonlinear transformation')
    elif i == 4: # animation
        ax[i].set_xlim([0,repeat_length])
        ax[i].autoscale_view()
        ax[i].set_ylim([-0.05,1.1])
        ax[i].set_xticklabels([0,1,2,3,4,5])
        im5, = plt.plot([], [], 'y-'); plt.title('spikes')


    ax[i].axes.get_xaxis().set_ticks([])


spikes = poisson.spiking(nonlinear)

def animate(n):
    x = range(n%repeat_length)
    im5.set_xdata(x)
    im5.set_ydata(spikes[0:n%repeat_length])

ani = anim.FuncAnimation(fig, animate, frames=len(spikes), interval=1, blit=False)

plt.show()
