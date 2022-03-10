"""
plotting_results_nb.py
====================================
tools for plotting the test results in the notebooks

note: this could probably be merged with the nb.py function?
    should move to utils in src?

"""

# import statements
import numpy
from matplotlib import pyplot as plt
import seaborn as sns


# Bland-altman like plots preparation
# maybe we should create a separate function for this?
def plot_bland_altman(ax, data, tolerances, tag, log_plot=False, xlim=None, ylim=None, label_xaxis=None, label_yaxis=None,
                    fig_title=None):

    g = sns.scatterplot(data=data, ax=ax, x=tag + '_ref', y='error_' + tag, hue='author',
                        hue_order=data.author.sort_values().unique(), style='author', size='author')

    tol = tolerances[tag]
    if not log_plot:
        # plot tolerance lines
        ax.axline((0, tol['atol']), slope=tol['rtol'], linestyle=":", color="slategray")
        ax.axline((0, -tol['atol']), slope=-tol['rtol'], linestyle=":", color="slategray")
    else:
        ax.set_xscale('log')  # logarithmic x-axis
        # plot tolerance lines
        if xlim != None:
            data_xrange = numpy.arange(xlim[0], xlim[1], 0.001 * xlim[1])
        else:
            data_xmax = data.max() + data.max() * 0.20
            data_xrange = numpy.arange(0, data_xmax, 0.001 * data_xmax)
            print(data_xmax)
            print(data_xrange)
        upper_limit = tol['atol'] + tol['rtol'] * data_xrange
        lower_limit = -tol['atol'] - tol['rtol'] * data_xrange
        ax.plot(data_xrange, upper_limit, data_xrange, lower_limit, linestyle=':', color='slategray')

    if ylim != None:
        ax.set_ylim(ylim[0], ylim[1])
    if xlim != None:
        ax.set_xlim(xlim[0], xlim[1])
    if label_xaxis != None:
        ax.set_xlabel(label_xaxis, fontsize=14)
    if label_yaxis != None:
        ax.set_ylabel(label_yaxis, fontsize=14)
    if fig_title != None:
        ax.set_title(fig_title, fontsize=16)

    #return g, ax
