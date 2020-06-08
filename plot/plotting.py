import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def dot_plot(x,y,loc = '.', name = 'plot.png'):
    plt.plot(x,y,'.b')
#    plt.xticks(ticks=axis1000[:10])
    plt.ylabel('real loss')
    plt.xlabel('predicted loss')
    plt.savefig(loc+'/' + name)
    plt.close()

if __name__=="__main__":
    import numpy as np
    x = np.array([1,2,3])
    y = np.array([5,2,4])
    y = [5,2,1]
    dot_plot(x,y)
