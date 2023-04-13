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
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns


def plot_bland_altman(
    ax,
    data,
    tolerances,
    tag,
    log_plot=False,
    xlim=None,
    ylim=None,
    label_xaxis=None,
    label_yaxis=None,
    fig_title=None,
):
    """
    this function creates bland-altman like plots including the tolerances

    :param ax: axis handles of the figure
    :param data: data-frame, where the difference between measured and reference values is already included as a column of the data frame
    :param tolerances: values for absolute and relative tolerances
    :param tag: the name of the parameter under investigation, e.g. r1 or conc
    :param log_plot: plot results on a log scale
    :param xlim, ylim: give limits for the x and/or y-axes
    :param label_xaxis, label_yaxis: label for x- or y-axis
    :param fig_title: give title for figure.

    citation: Bland JM and Altman DG "Measuring agreement in method comparison studies" Statistical Methods in Medical Research 1999; 8: 135-160
    """

    g = sns.scatterplot(
        data=data,
        ax=ax,
        x=tag + "_ref",
        y="error_" + tag,
        hue="author",
        hue_order=data.author.sort_values().unique(),
        style="author",
        s=120,
    )

    tol = tolerances[tag]
    if not log_plot:
        # plot tolerance lines
        ax.axline((0, tol["atol"]), slope=tol["rtol"], linestyle=":", color="slategray")
        ax.axline(
            (0, -tol["atol"]), slope=-tol["rtol"], linestyle=":", color="slategray"
        )
    else:
        ax.set_xscale("log")  # logarithmic x-axis
        # plot tolerance lines
        if xlim != None:
            data_xrange = numpy.arange(xlim[0], xlim[1], 0.001 * xlim[1])
        else:
            data_xmax = data.max() + data.max() * 0.20
            data_xrange = numpy.arange(0, data_xmax, 0.001 * data_xmax)
            print(data_xmax)
            print(data_xrange)
        upper_limit = tol["atol"] + tol["rtol"] * data_xrange
        lower_limit = -tol["atol"] - tol["rtol"] * data_xrange
        ax.plot(
            data_xrange,
            upper_limit,
            data_xrange,
            lower_limit,
            linestyle=":",
            color="slategray",
        )

    if ylim != None:
        ax.set_ylim(ylim[0], ylim[1])
    if xlim != None:
        ax.set_xlim(xlim[0], xlim[1])
    if label_xaxis != None:
        ax.set_xlabel(label_xaxis)
    if label_yaxis != None:
        ax.set_ylabel(label_yaxis)
    if fig_title != None:
        ax.set_title(fig_title)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")


def bland_altman_statistics(data, par, grouptag):
    """
    this function determines bias and limits of agreement based on bland-altman statistics

    :param data: a pandas dataframe
    :param par: is the name of the column in data for which the bias etc needs to be calculated.
        This can be for example the difference between measured and reference values or the ratio of measured and
        reference values. See citation for more info.
    :param grouptag: is the label for which the data needs to be grouped before calculating the bias
    :return: resultsBA: bias, standard deviation, lower and upper limits of agreement

    citation: Bland JM and Altman DG "Measuring agreement in method comparison studies" Statistical Methods in Medical Research 1999; 8: 135-160
    """

    subset_data = data[[grouptag, par]]

    # calculate mean error = bias; this is done per group, defined in grouptag
    bias = subset_data.groupby(grouptag).mean()
    bias.rename(columns={par: "bias"}, inplace=True)

    # calculate std for lower limits of agreement
    stdev = subset_data.groupby(grouptag).std()
    stdev.rename(columns={par: "stdev"}, inplace=True)
    resultsBA = bias.join(stdev)

    resultsBA["LoA lower"] = resultsBA["bias"] - 1.96 * resultsBA["stdev"]
    resultsBA["LoA upper"] = resultsBA["bias"] + 1.96 * resultsBA["stdev"]

    return resultsBA


def make_catplot(x, y, data, ylabel, **plotopts):
    g = sns.catplot(x=x, y=y, data=data, **plotopts)
    g.set_titles(col_template="{col_name}")
    g.set(xlabel=None)
    g.set_ylabels(ylabel, clear_inner=False)
    g.set_xticklabels(rotation=45, fontsize=16, ha="right", rotation_mode="anchor")
    g._legend.set_title("ContributionID")
    # Add vertical add midpoints between each major tick
    for ax in g.axes.flatten():
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(which="minor", axis="x", linestyle=":")
        # Hide the minor ticks (distracting) by making it white
        ax.tick_params(axis="x", which="minor", colors="white")
    plt.show()  # Could also do `return g` for more flexibility
