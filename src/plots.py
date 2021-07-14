import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.cm as cm

def class_counts(y):
    labels, labels_inverse, label_counts = np.unique(y, return_inverse=True, return_counts=True)
    return labels, labels_inverse, label_counts

def class_bar_plot(ax, y, title, xlabel, ylabel, plot_type='v', bar_color ='turquoise'):
    '''Plot a bar chart with the value labeled for each bar.
    
    Parameters
    ----------
    ax: axes object
        axes for plot
    y: array
        label array to be plotted
    plot_type: str
        'v' for vertical bar chart, 'h' for horizontal bar chart
    title: str
        title of plot
    xlabel: str
        x label for plot
    ylabel: str
        y label for plot
    bar_color: str
        desired bar color
    
    Returns
    -------
    None
    '''
    labels, _, label_counts = class_counts(y)
    if plot_type == 'h':
        bars = ax.barh(labels, label_counts, color = bar_color)
        
        # Add labels for each bar
        for bar in bars:
          width = bar.get_width()
          label_y_pos = bar.get_y() + bar.get_height() / 2
          ax.text(width, label_y_pos, s=f'{width}', va='center')
    else:
        bars = ax.bar(labels, label_counts, color = bar_color)
        
        # Add labels for each bar
        for bar in bars:
          height = bar.get_height()
          label_x_pos = bar.get_x() + bar.get_width() / 2
          ax.text(label_x_pos, height, s=f'{height}', ha='center')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

if __name__ == '__main__':
    train, test = np.loadtxt('data/sat.trn'), np.loadtxt('data/sat.tst')
    y_train, y_test = train[:,-1], test[:,-1]

    fig, ax =plt.subplots()
    class_bar_plot(ax, y_test, 'Occurrences of Each Class in Test Set', 'Class', 'Number of Occurrences')
    ax.set_xticklabels(['','1 - red soil', '2 - cotton crop', '3 - grey soil', '4 - damp grey soil', '5 - soil w/veg.', '6 - mixture', '7 - very damp grey soil'], rotation=45)
    plt.tight_layout()
    plt.show()


    