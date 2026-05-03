import numpy as np
import matplotlib.pyplot as plt

ic_comps = np.load("new_ics.npy")
decisions = []
for i in range(ic_comps.shape[1]):
    fig, ax = plt.subplots(figsize=(20,5))
    ax.plot(ic_comps[:,i], color='silver')
    ax.vlines(x=[562,675,1103,1963,2162,2295,2296,2297,472,473,478,447,697,1020,2581,2600],ymin=ic_comps.T[i,:].min(),ymax=ic_comps.T[i,:].max(),ls='--',color='g',lw = .7)
    plt.title(f'IC comp {i}')
    try:
        zoom_inf, zoom_sup = plt.ginput(2)
        x_inf, _ = zoom_inf
        if x_inf < 0:
            x_inf = 0
        x_sup, _ = zoom_sup
        if x_sup > len(ic_comps[:,i]):
            x_sup = len(ic_comps[:,i])
        plt.close('all')
        #  Second plot with the zoomed window to select precisely beginning and end of bout
        fig, ax = plt.subplots(figsize=(20, 5))
        plt.plot(ic_comps[:,i][int(x_inf):int(x_sup)], 'silver')
    except ValueError:
        pass

    #  Ask user for manual category
    plt.close('all')
    plt.figure(figsize=(4, 2))
    plt.text(1, 0.5, 'noise', fontsize=17, color='white', fontfamily='sans-serif',
             bbox=dict(boxstyle='round', facecolor='magenta', alpha=0.5))
    plt.text(0.5, 0.5, 'signal', fontsize=17, color='white', fontfamily='sans-serif',
             bbox=dict(boxstyle='round', facecolor='royalblue', alpha=0.5))
    plt.xlim(0.2, 1.2)
    plt.ylim(0.2, 0.8)
    plt.title('Choose manual cat')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    x, y = plt.ginput(1)[0]
    if x <= 0.7:
        decisions.append('signal')
    else:
        decisions.append('noise')
