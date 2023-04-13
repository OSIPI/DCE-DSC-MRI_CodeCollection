"""
Authors: Jesper Browall, Jonathan Arvidsson, Sahlgrenska University Hospital and Gothenburg University, Gothenburg, Sweden

This file contains plot functions called by the class AutoAifSel when input variable 'doPlot' is set to True.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from cycler import cycler
import numpy as np
from os.path import join


def get_plot_idcs(unique_slices, num_slices):
    if num_slices >= len(unique_slices):
        return unique_slices
    else:
        strt_idx = (len(unique_slices) - num_slices) // 2
        end_idx = strt_idx + num_slices
        return unique_slices[strt_idx:end_idx]


def get_windowlevels(data, percent, num_bins):
    histogram = np.histogram(data, bins=num_bins)
    idcs = histogram[0] >= histogram[0].sum() * (percent / 100) / 2
    idcs = np.append(idcs, False)  # Add a False value to get correct length on vector
    min_val = histogram[1][idcs].min()
    max_val = histogram[1][idcs].max()

    return [min_val, max_val]


def get_figure_settings(do_Mouridsen2006, numSlices):
    # Some specific settings depending on we're going with Mourdisen2006 or our own method.
    # The only real difference is figure height/rows and colormaps.
    widthList = [4]
    widthList.extend([1] * numSlices)

    if do_Mouridsen2006:
        figure = plt.subplots(
            ncols=1 + numSlices,
            nrows=7,
            constrained_layout=True,
            figsize=(6.4 + 1.6 * numSlices, 14),
            gridspec_kw={
                "width_ratios": widthList,
                "height_ratios": [1, 1, 1, 1, 1, 1, 1],
            },
        )

        titles = [
            "Concentration data in AIF-ROI",
            "The largest 10% AUC of remaining vxls within ROI",
            "Removed largest 25 % ireg of entire brain",
            "Cluster result 1",
            "Cluster result 2",
            "AIF in relation to scaled cluster centers",
            "Final AIF",
        ]

        cluster_colmap_plot = [
            (1, 0, 0, 1),
            (0, 1, 0, 1),
            (0, 0, 1, 1),
            (0, 1, 1, 1),
            (1, 0, 1, 1),
        ]
        cluster_colmap_vals_disp = np.array(
            [
                [1, 1, 1, 0],
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [1, 0, 1, 1],
            ]
        )
        cluster_colmap = ListedColormap(cluster_colmap_vals_disp)

        clustercenter_colmap_plot = [
            (1, 215 / 255, 0, 1),
            (1, 0, 0, 1),
            (0, 1, 0, 1),
            (0, 0, 1, 1),
            (0, 1, 1, 1),
            (1, 0, 1, 1),
        ]
        clustercenter_colmap_vals_disp = np.array(
            [[1, 1, 1, 0], np.array(clustercenter_colmap_plot)[0]]
        )
        clustercenter_colmap = ListedColormap(clustercenter_colmap_vals_disp)

    else:
        figure = plt.subplots(
            ncols=1 + numSlices,
            nrows=6,
            constrained_layout=True,
            figsize=(6.4 + 1.6 * numSlices, 12),
            gridspec_kw={
                "width_ratios": widthList,
                "height_ratios": [1, 1, 1, 1, 1, 1],
            },
        )

        titles = [
            "Concentration data in AIF-ROI",
            "The largest 20% AUC of remaining vxls within ROI",
            "Removed largest 1 % ireg of entire brain",
            "Cluster result",
            "AIF in relation to scaled cluster centers",
            "Final scaled AIF",
        ]

        cluster_colmap_plot = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
        cluster_colmap_vals_disp = np.array(
            [[1, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        )
        cluster_colmap = ListedColormap(cluster_colmap_vals_disp)

        clustercenter_colmap_plot = [
            (1, 215 / 255, 0, 1),
            (1, 0, 0, 1),
            (0, 1, 0, 1),
            (0, 0, 1, 1),
        ]
        clustercenter_colmap_vals_disp = np.array(
            [[1, 1, 1, 0], np.array(clustercenter_colmap_plot)[0]]
        )
        clustercenter_colmap = ListedColormap(clustercenter_colmap_vals_disp)

    # Make colormap
    colmaps_cluster_result = {
        "plotColors": cluster_colmap_plot,
        "dispcolmap": cluster_colmap,
    }

    colmaps_cluster_centers = {
        "plotColors": clustercenter_colmap_plot,
        "dispcolmap": clustercenter_colmap,
    }

    col_AIF = [(1, 215 / 255, 0, 1)]
    colmaps_AIF = {"plotColors": col_AIF, "dispcolmap": clustercenter_colmap}

    return figure, titles, colmaps_cluster_result, colmaps_cluster_centers, colmaps_AIF


def plotConcData(
    figure,
    xData,
    conc_data,
    dispData,
    ireg_data,
    ROI,
    plot_slices,
    colmap,
    colmapnorm,
    normtype,
    windowLevels,
    x_lim,
    titles,
    callCount,
):
    """
    Function used to plot the concentration data within a given ROI. The colormap goes from black to white.
    With black and white being the curves with lowest and highest area under curve (AUC), respectively.
    It returns the figure object and the callCount variable

    figure: The figure object to be generated
    xData: 1D numpy array with time points of the concentration data
    conc_data: 4D numpy array  with the conctration data of interest
    dispData: 3D numpy array with same dimensions as conc_data in x,y and z direction. dispData is used to display where
    the ROI is located aswell as the intensity map of the AUC-value
    ROI: 3D numpy array with the region of interes. Must have same dimension as dispData
    plot_slices: 1D numpy array with integers indicating where the ROI has values of 1
    colmap: Matplotlib colormap used for plotting conc_data
    colmapnorm: Matplotlib colormap norm used to get same color for same AUC throughout the three first plots
    windowLevels: Windowlevles for the dispData, may have to be adjusted for other kinds of data.
    x_lim: Limits to i x-axis in the plot
    titles: List of lenth 6 with the different titles of the plots
    callCount: Variable that keeps track of which subplot to edit

    """
    mycmap = colmap  # Creating a custom cmap for the dispdata with the first color totally transparent
    cmaplist = [colmap(i) for i in range(mycmap.N)]
    cmaplist[0] = (1.0, 1.0, 1.0, 0.0)
    mycmap = mycmap.from_list("Custom cmap", cmaplist, mycmap.N)

    ROI_4D = np.repeat(ROI[:, :, :, np.newaxis], np.size(conc_data, 3), axis=3)

    conc_data_temp = np.zeros(np.shape(ROI_4D))
    conc_data_temp[ROI_4D] = conc_data[ROI_4D]
    conc_data_temp_resh = np.reshape(
        conc_data_temp,
        (
            np.shape(conc_data_temp)[0]
            * np.shape(conc_data_temp)[1]
            * np.shape(conc_data_temp)[2],
            np.shape(conc_data_temp)[3],
        ),
    )
    conc_data_plot = conc_data_temp_resh[
        ~np.all(conc_data_temp_resh == 0, axis=1)
    ]  # Removes all rows with only zeros

    conc_intensity = (
        np.zeros(np.shape(ROI)) + colmapnorm.vmin - 1
    )  # Putting a background level at one under the lowest in the entire ROI
    colorScalarMappable = cm.ScalarMappable(norm=colmapnorm, cmap=colmap)

    # Down sample the data to be maximum 500 curves
    if np.shape(conc_data_plot)[0] > 500:
        downSampfact = int(np.round(np.shape(conc_data_plot)[0] / 500))
        conc_data_plot_down = np.take(
            conc_data_plot, range(0, np.shape(conc_data_plot)[0], downSampfact), axis=0
        )
    else:
        conc_data_plot_down = np.array(conc_data_plot)
        downSampfact = 1

    if normtype == "auc":
        # Setting up a custom color cycle to reflect the AUC of the voxel
        x = np.sum(conc_data_plot_down, axis=1)
        conc_intensity[ROI] = np.sum(conc_data_plot, axis=1)

    if normtype == "ireg":
        # Setting up a custom color cycle to reflect the AUC of the voxel
        ireg1D = np.reshape(ireg_data[ROI], ROI.sum())
        conc_intensity[ROI] = ireg1D
        x = np.take(ireg1D, range(0, np.shape(conc_data_plot)[0], downSampfact))

    custom_cycler = cycler(color=list(np.array(colorScalarMappable.to_rgba(x))))

    figure[1][callCount][0].set_prop_cycle(custom_cycler)
    figure[1][callCount][0].plot(xData, conc_data_plot_down.T, linewidth=0.1)
    figure[1][callCount][0].set_title(titles[callCount])
    figure[1][callCount][0].set_facecolor("lightgray")
    figure[1][callCount][0].set_xlim(x_lim)

    for n, sl in enumerate(plot_slices):
        figure[1][callCount][1 + n].imshow(
            np.rot90(dispData[:, :, sl]),
            "gray",
            interpolation="none",
            vmin=windowLevels[0],
            vmax=windowLevels[1],
        )
        figure[1][callCount][1 + n].imshow(
            np.rot90(conc_intensity[:, :, sl]),
            norm=colmapnorm,
            cmap=mycmap,
            interpolation="none",
            alpha=0.9,
        )

    callCount = callCount + 1

    return figure, callCount


def plotClusterResult(
    figure,
    xData,
    yData,
    dispData,
    ROI,
    plot_slices,
    colmaps,
    windowLevels,
    x_lim,
    titles,
    callCount,
    k_means,
):
    """
    Function used to plot the result from k means clustering. The cluster center with lowest first moment gets the color red, 2nd lowest
    gets color green och highest first momment gets color blue.


    figure: The figure object to be generated
    xData: 2D numpy array with time points to be plotted. Each column represents one curve to be plotted.
    yData: 2D numpy array with the concentration that was used in the k means clustering. Each row represents a concentration time curve.
    dispData: 3D numpy array with same dimensions as conc_data in x,y and z direction. dispData is used to display where
    the ROI is located aswell as the intensity map of the AUC-value
    ROI: 3D numpy array with the region of interes. Must have same dimension as dispData
    plot_slices: 1D numpy array with integers indicating where the ROI has values of 1
    colmaps: A dictionary with colormaps.
    colmaps['plotColors'] is a list of tuples which creates a RGB colormap
    colmaps['dispcolmap'] is a matplotlib colormap which creates a RGB colormap
    windowLevels: Windowlevles for the dispData, may have to be adjusted for other kinds of data.
    x_lim: Limits to i x-axis in the plot
    titles: List of lenth 6 with the different titles on the plots
    callCount: Variable that keeps track of which subplot to edit
    k_means: Result from k-means clustering

    """
    num_clust = np.shape(k_means.cluster_centers_)[0]
    # Getting the cluster result and puts in in a 3D matrix
    labelVol = (
        np.zeros(np.shape(ROI)) - 1
    )  # We'll be filling with zeros so no zeros in the volume
    labelVol[ROI] = k_means.labels_
    ROI_4D = np.repeat(ROI[:, :, :, np.newaxis], np.size(yData, 1), axis=3)

    # 4d matrix with the concetntration data
    yData4D = np.zeros(
        (np.shape(ROI)[0], np.shape(ROI)[1], np.shape(ROI)[2], np.shape(yData)[1])
    )
    yData4D[ROI_4D] = np.reshape(yData, (np.sum(ROI_4D)))

    # Order the clustercenter based on first moment
    ClusterOrder = np.sum(
        xData.T[:num_clust, :] * k_means.cluster_centers_, axis=1
    ).argsort()
    clusterDict = {}

    for i in range(num_clust):
        clusterDict["cluster" + str(i)] = yData4D[labelVol == ClusterOrder[i]]

    # Creating a 3D matrix with value 1 for the cluster with lowest first moment, value 2 for next lowest etc
    clusterDistrib = np.zeros(np.shape(ROI))

    # Creating a temporary array with the cluster distribution
    tmp = np.zeros(np.shape(k_means.labels_))
    for n in range(num_clust):
        tmp = tmp + (k_means.labels_ == ClusterOrder[n]) * (n + 1)

    clusterDistrib[ROI] = tmp
    normalize = mpl.colors.Normalize(vmin=0, vmax=num_clust)

    for col, cl in zip(colmaps["plotColors"], clusterDict.keys()):
        figure[1][callCount][0].plot(xData, clusterDict[cl].T, color=col, linewidth=0.1)

    for n, sl in enumerate(plot_slices):
        figure[1][callCount][1 + n].imshow(
            np.rot90(dispData[:, :, sl]),
            "gray",
            interpolation="none",
            vmin=windowLevels[0],
            vmax=windowLevels[1],
        )
        figure[1][callCount][1 + n].imshow(
            np.rot90(clusterDistrib[:, :, sl]),
            norm=normalize,
            cmap=colmaps["dispcolmap"],
            interpolation="none",
            alpha=0.9,
        )

    figure[1][callCount][0].set_facecolor("lightgray")
    figure[1][callCount][0].set_xlim(x_lim)
    figure[1][callCount][0].set_title(titles[callCount])

    callCount = callCount + 1
    return figure, callCount


def plotClusterData(
    figure,
    xData,
    yData,
    dispData,
    ROI,
    plot_slices,
    colmaps,
    windowLevels,
    x_lim,
    titles,
    num_clust,
    callCount,
):
    """
    Function used to plot the cluster centers and the final AIF selection.


    figure: The figure object to be generated
    xData: 2D numpy array with time points to be plotted. Each column represents one curve to be plotted.
    yData: 2D numpy array with the concentration that was used in the k means clustering. Each row represents a concentration time curve.
    dispData: 3D numpy array with same dimensions as conc_data in x,y and z direction. dispData is used to display where
    the ROI is located aswell as the intensity map of the AUC-value
    ROI: 3D numpy array with the region of interes. Must have same dimension as dispData
    plot_slices: 1D numpy array with integers indicating where the ROI has values of 1
    colmaps: A dictionary with colormaps.
    colmaps['plotColors'] is a list of tuples which creates a RGB colormap
    colmaps['dispcolmap'] is a matplotlib colormap which creates a RGB colormap
    windowLevels: Windowlevles for the dispData, may have to be adjusted for other kinds of data.
    x_lim: Limits to i x-axis in the plot
    titles: List of lenth 6 with the different titles on the plots
    callCount: Variable that keeps track of which subplot to edit
    """
    LW = [3]
    LW.extend(np.ones(num_clust))
    labels = [
        "AIF",
        "Cluster lowest FM",
        "Cluster 2nd lowest FM",
        "Cluster 3rd lowest FM",
        "Cluster 4th lowest FM",
        "Cluster 5th lowest FM",
    ]
    labels = labels[: num_clust + 1]
    for y, col in enumerate(colmaps["plotColors"]):
        figure[1][callCount][0].plot(
            xData, yData[y, :], color=col, linewidth=LW[y], label=labels[y]
        )

    for n, sl in enumerate(plot_slices):
        figure[1][callCount][1 + n].imshow(
            np.rot90(dispData[:, :, sl]),
            "gray",
            interpolation="none",
            vmin=windowLevels[0],
            vmax=windowLevels[1],
        )
        figure[1][callCount][1 + n].imshow(
            np.rot90(ROI[:, :, sl]),
            cmap=colmaps["dispcolmap"],
            interpolation="none",
            alpha=0.9,
        )

    figure[1][callCount][0].legend(prop={"size": 6}, loc="best")
    figure[1][callCount][0].set_facecolor("lightgray")
    figure[1][callCount][0].set_xlim(x_lim)
    figure[1][callCount][0].set_title(titles[callCount])

    callCount = callCount + 1
    return figure, callCount


def plot_figures(self):
    conc_data = np.array(self.conc_data)
    peakconc_data = np.array(self.peakconc_data)
    ROI_data = np.array(self.ROI_data)
    clustIdcs = range(self.clustIdcs.start, self.clustIdcs.stop)
    AIF = np.array(self.AIF)
    clusterOrder = np.array(self.clusterOrder_finalClust)

    if self.plot_config == -1:
        plot_slices = np.unique(np.nonzero(self.ROI_AIF)[2])
        numSlices = len(plot_slices)

    elif self.plot_config == 0 or self.plot_config >= len(
        np.unique(np.nonzero(self.ROI_data)[2])
    ):
        plot_slices = np.unique(np.nonzero(self.ROI_data)[2])
        numSlices = len(plot_slices)

    else:
        numSlices = self.plot_config
        unique_slices = np.unique(np.nonzero(self.ROI_data)[2])
        plot_slices = get_plot_idcs(unique_slices, numSlices)

    (
        figure,
        titles,
        colmaps_cluster_result,
        colmaps_cluster_centers,
        colmaps_AIF,
    ) = get_figure_settings(self.do_Mouridsen2006, numSlices)

    colmap = cm.hot
    x_lim = [-1, np.shape(conc_data)[-1] + 1]
    # Normalizes the colormap to the min/max of the AUC of the concData and plots the curves
    colmap_limits_auc = get_windowlevels(np.sum(conc_data[ROI_data], axis=1), 1, 50)
    colmapnorm_auc = mpl.colors.Normalize(colmap_limits_auc[0], colmap_limits_auc[1])

    # Normalizes the colormap to the min/max of the Ireg-measure of the concData and plots the curves
    colmap_limits_ireg = get_windowlevels(self.ireg_data[self.ROI2], 10, 500)
    colmapnorm_ireg = mpl.colors.Normalize(colmap_limits_ireg[0], colmap_limits_ireg[1])

    if self.do_Mouridsen2006:
        figure[0].suptitle(
            "auto_AIF_result_fig_Mouridsen" + self.plot_suffix, fontsize=16
        )
    else:
        figure[0].suptitle("auto_AIF_result_fig" + self.plot_suffix, fontsize=16)

    xData = np.array(range(0, np.shape(conc_data)[3]))
    windowLevels = get_windowlevels(
        np.reshape(peakconc_data[ROI_data], ROI_data.sum()), 1, 50
    )
    callCount = 0

    # -------------- Plot result from removing irrelevant vxls------------------
    figure, callCount = plotConcData(
        figure,
        xData[clustIdcs],
        conc_data[:, :, :, clustIdcs],
        peakconc_data,
        self.ireg_data,
        ROI_data,
        plot_slices,
        colmap,
        colmapnorm_auc,
        "auc",
        windowLevels,
        x_lim,
        titles,
        callCount,
    )
    figure, callCount = plotConcData(
        figure,
        xData[clustIdcs],
        conc_data[:, :, :, clustIdcs],
        peakconc_data,
        self.ireg_data,
        self.ROI2,
        plot_slices,
        colmap,
        colmapnorm_ireg,
        "ireg",
        windowLevels,
        x_lim,
        titles,
        callCount,
    )
    figure, callCount = plotConcData(
        figure,
        xData[clustIdcs],
        conc_data[:, :, :, clustIdcs],
        peakconc_data,
        self.ireg_data,
        self.ROI3,
        plot_slices,
        colmap,
        colmapnorm_auc,
        "auc",
        windowLevels,
        x_lim,
        titles,
        callCount,
    )

    # ------------------ Plot result from k_means--------------------------

    if (
        self.do_Mouridsen2006
    ):  # An extra plot when we have two cluster results to present
        t_rep4 = np.repeat(
            self.t4[:, np.newaxis], np.shape(self.k_means_preClust.labels_)[0], axis=1
        )
        figure, callCount = plotClusterResult(
            figure,
            t_rep4,
            self.concData2DNorm_preClust,
            peakconc_data,
            self.ROI3,
            plot_slices,
            colmaps_cluster_result,
            windowLevels,
            x_lim,
            titles,
            callCount,
            self.k_means_preClust,
        )

    t_rep5 = np.repeat(
        self.t4[:, np.newaxis], np.shape(self.k_means_finalClust.labels_)[0], axis=1
    )

    figure, callCount = plotClusterResult(
        figure,
        t_rep5,
        self.concData2DNorm_finalClust,
        peakconc_data,
        self.ROI_preClust,
        plot_slices,
        colmaps_cluster_result,
        windowLevels,
        x_lim,
        titles,
        callCount,
        self.k_means_finalClust,
    )

    sFacDisp = float(
        np.max(AIF[clustIdcs]) / np.max(self.k_means_finalClust.cluster_centers_)
    )  # Scalefactor so the plots have same max amplitude, just for estetics

    cluster_centers_list = [
        sFacDisp * self.k_means_finalClust.cluster_centers_[clusterOrder[i], :]
        for i in range(self.numClust)
    ]

    AIF_plus_clusterCenters = np.append([AIF[clustIdcs]], cluster_centers_list, axis=0)
    figure, callCount = plotClusterData(
        figure,
        self.t4,
        AIF_plus_clusterCenters,
        peakconc_data,
        self.ROI_AIF,
        plot_slices,
        colmaps_cluster_centers,
        windowLevels,
        x_lim,
        titles,
        self.numClust,
        callCount,
    )

    t6 = np.array(range(0, len(AIF)))
    figure, callCount = plotClusterData(
        figure,
        t6,
        np.array([AIF]),
        peakconc_data,
        self.ROI_AIF,
        plot_slices,
        colmaps_AIF,
        windowLevels,
        x_lim,
        titles,
        self.numClust,
        callCount,
    )

    if self.do_Mouridsen2006:
        fileName = "auto_AIF_result_fig_Mouridsen" + self.plot_suffix + ".jpeg"
    else:
        fileName = "auto_AIF_result_fig" + self.plot_suffix + ".jpeg"

    figure[0].savefig(join(self.outputDir, fileName), dpi=300)
    figure[0].clf()
    plt.close(figure[0])
