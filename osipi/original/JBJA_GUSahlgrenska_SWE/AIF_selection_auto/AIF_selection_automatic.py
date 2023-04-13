"""
Authors: Jesper Browall, Jonathan Arvidsson, Sahlgrenska University Hospital and Gothenburg University, Gothenburg, Sweden

The file AIF_selection_automatic contains the fully automatic whole-brain AIF-selection method for DSC-MRI concentration data
orignially published by Mouridsen et al. 2006 titled "Automatic selection of arterial input function using cluster analysis"
(DOI: 10.1002/mrm.20759).

The file AIF_selection_automatic also includes a semi-automatic implementation, similar to the original version, in which a
user defined ROI is provided to guarantee the spatial origin of the resulting AIF. This version is reached by setting the
second input variable to False.

How to use:
AIF_vec, AIF_ROI = AutoAifSel(conc_fn, brain_mask_fn, outputDir, plot_config, do_Mouridsen2006, ROI_fn).return_vals()

Where:
conc_fn*        is a filepath to a .nii(.gz)-file containing 4D concentration data (DSC-MRI)
brain_mask_fn*  is a filepath to a .nii(.gz)-file containing a whole-brain (skull stripping) binary/boolean mask
outputDir       is the output directory where plots will be saved, if empty the current working directory will be used
plotconfig      is an integer to specify if and how many slices to include in AIF-selection plot. Choosing to many dramatically increases time to save the image.
                If an integer greater than 0 is selected, it will choose slices from the middle of ROI/brainMask and work towards the edges.
                If set to at least -1 it will save a .jpg-file with the corresponding figure.
                Default value is -2

               -2: No plot
               -1: All slices with resulting AIF
                0: All avalailable slices i ROI/brainMask
                1: One slice
                2: Two slices
                3: etc..
plot_suffix     is a string to add a suffix to the resulting figure if saved
                Default value is ''
do_Mouridsen2006 is a boolean. True = do AIF selecten based on Mouridsen2006, False = do our ajusted method
                 default value is True
ROI_fn**        is a filepath to a .nii(.gz)-file containing a binary/boolean mask of area to select AIF from including
                the desired artery and surrounding parenchyma, artery/parenchyma volume quote approx (30/70)

AIF_vec         is the resulting AIF vector saved as numpy 1D array
AIF_ROI         is a mask containing the voxels from which the AIF is calculated, saved as a 3D numpy array

*  Mandatory
** Mandatory if do_Mouridsen2006 = False. If do_Mouridsen2006 = True it will be set to brain_mask_fn
"""


import nibabel as nib
from os.path import join, isfile
from os import getcwd
import numpy as np
import csv
from scipy import stats
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
import pandas as pd
import sys
from plots import plot_figures


class AutoAifSel:
    def __init__(
        self,
        conc_fn,
        brain_mask_fn,
        outputDir=getcwd(),
        plot_config=2,
        plot_suffix="",
        do_Mouridsen2006=True,
        ROI_fn="",
    ):
        # Ta ställning till sållning av IREG och AUC -- GÖR ETT TEST!!! GÅr inte!!!

        self.conc_fn = conc_fn
        self.brain_mask_fn = brain_mask_fn
        self.pre_b_idx = 1
        self.post_b_idx = 12
        self.outputDir = outputDir
        self.plot_suffix = plot_suffix

        if do_Mouridsen2006:
            self.ROI_fn = str(brain_mask_fn)

        else:
            self.ROI_fn = ROI_fn

        self.plot_config = plot_config
        self.do_Mouridsen2006 = do_Mouridsen2006
        self.check_data()
        self.load_data()
        self.remove_non_finite()

        if do_Mouridsen2006:
            self.numClust = 5
            self.remove_irrelevant_vxls()
            self.pre_proc()
            self.clustercount = 1
            (
                self.k_means_preClust,
                self.concDatalowfM_preClust,
                self.concData2DNorm_preClust,
                self.ROI_preClust,
                _,
            ) = self.clustering()
            (
                self.k_means_finalClust,
                self.concDatalowfM_finalClust,
                self.concData2DNorm_finalClust,
                self.ROI_finalClust,
                self.clusterOrder_finalClust,
            ) = self.clustering()
            self.ROI_AIF = self.ROI_finalClust
            self.calc_aif()

        else:
            self.numClust = 3
            self.remove_irrelevant_vxls()
            self.pre_proc()
            self.clustercount = 1
            self.ROI_preClust = (
                self.ROI3
            )  # Since we only cluster once, ROI3 will be equivalent with ROI_preClust in Mouridsen2006
            (
                self.k_means_finalClust,
                self.concDatalowfM_finalClust,
                self.concData2DNorm_finalClust,
                self.ROI_finalClust,
                self.clusterOrder_finalClust,
            ) = self.clustering()
            self.calc_aif()

        if self.plot_config > -2:
            plot_figures(self)

        self.return_vals()

    def check_data(self):
        if not isfile(self.conc_fn):
            print("The concentration file doesnt exist, script exiting")
            sys.exit()

        if not isfile(self.brain_mask_fn):
            print("The brain mask file doesnt exist, script exiting")
            sys.exit()

        if not self.do_Mouridsen2006:
            if not isfile(self.ROI_fn):
                print("The ROI-file doesnt exist, script exiting")
                sys.exit()

    def load_data(self):
        # Loads the data into arrays except för b_idx which is just an int.
        self.ROI_data = nib.load(self.ROI_fn).get_data().astype(bool)
        self.brain_mask_data = nib.load(self.brain_mask_fn).get_data().astype(bool)
        self.conc_data = nib.load(self.conc_fn).get_data()

        self.b_idx = self.calc_b_idx()
        self.firstM_data = self.calc_features("firstM_feat")
        self.ireg_data = self.calc_features("ireg_feat")
        self.auc_data = self.calc_features("auc_feat")
        self.peakconc_data = self.calc_peakconc()

    def calc_b_idx(self):
        # The b_idx is calculated via an in house script that checks for difference in concentration as a mean over entire volume.
        # You can easily generate a plot to se the result with the code below

        # Remove non-finite data
        conc_data_finite = np.zeros(np.shape(self.conc_data))
        conc_data_finite[np.isfinite(self.conc_data)] = self.conc_data[
            np.isfinite(self.conc_data)
        ]
        conc_mean = np.sum(conc_data_finite, axis=(0, 1, 2))

        for idx in range(0, len(conc_mean)):
            diff1 = (conc_mean[idx] - conc_mean[idx + 1]) / (
                conc_mean[idx] - np.max(conc_mean)
            )
            diff2 = (conc_mean[idx] - conc_mean[idx + 2]) / (
                conc_mean[idx] - np.max(conc_mean)
            )
            if diff1 > 0.07 and diff2 > 0.15 and idx > 5:
                b_idx = idx
                break
        # Code for displaying b_idx
        """import matplotlib.pyplot as plt
        plt.plot(range(0, np.shape(self.conc_data)[3]), conc_mean)
        plt.plot(b_idx, conc_mean[b_idx], '*', label = 'b_idx')
        plt.legend()
        plt.show()"""

        return b_idx

    def calc_features(self, ftype):
        # Calculates the features and removes non-finite values
        if ftype == "auc_feat":
            feature = np.sum(self.conc_data[:, :, :, self.b_idx - 2 :], axis=3)

        if ftype == "ireg_feat":
            secondDeriv = np.diff(self.conc_data, n=2, axis=3)
            feature = np.sum(np.power(secondDeriv, 2), axis=3)

        if ftype == "firstM_feat":
            timeVec = range(
                1, np.size(self.conc_data, 3) - self.b_idx + 1
            )  # Tidsvektor med N-b_idx element
            feature = np.sum(self.conc_data[:, :, :, self.b_idx :] * timeVec, axis=3)

        finiteMask = np.zeros(np.shape(self.conc_data)[0:3]).astype(bool)
        finiteMask[np.isfinite(feature)] = True
        feature_z = np.zeros(np.shape(self.conc_data)[0:3])
        feature_z[self.brain_mask_data * finiteMask] = stats.zscore(
            feature[self.brain_mask_data * finiteMask]
        )
        return feature_z

    def calc_peakconc(self):
        idx_max = np.where(
            self.conc_data == np.amax(self.conc_data[np.isfinite(self.conc_data)])
        )[0][0]
        peakconc_data = np.mean(
            self.conc_data[:, :, :, self.b_idx + 1 : idx_max + 3], 3
        )
        return peakconc_data

    def remove_non_finite(self):
        self.peakconc_data[
            np.isfinite(self.peakconc_data) == False
        ] = 0  # This is for displaying only, Nans makes it look wierd.

    def remove_irrelevant_vxls(self):
        # ---------------------------------------------------------
        # |  Save largest (100-P_auc) % auc of remaining vxls within ROI, save it as ROI2   |
        # ---------------------------------------------------------
        if self.do_Mouridsen2006:
            P_auc = 90

        else:
            P_auc = 80

        Smallest_auc = np.percentile(self.auc_data[self.ROI_data], P_auc)
        ROI2 = (self.auc_data >= Smallest_auc) * self.ROI_data

        # ---------------------------------------------------------
        # |        Remove largest (100-P_ireg) % ireg of entire brain, save new ROI in ROI3   |
        # ---------------------------------------------------------
        if self.do_Mouridsen2006:
            P_ireg = 75

        else:
            P_ireg = 99

        Largest_ireg = np.percentile(self.ireg_data[self.brain_mask_data], P_ireg)
        ROI3 = (self.ireg_data <= Largest_ireg) * ROI2

        self.ROI2 = ROI2
        self.ROI3 = ROI3
        self.ROI3_4D = np.repeat(
            ROI3[:, :, :, np.newaxis], np.size(self.conc_data, 3), axis=3
        )

    def pre_proc(self):
        b_idx = int(self.b_idx)
        pre_b_idx = int(self.pre_b_idx)
        post_b_idx = int(self.post_b_idx)
        conc_data = np.array(self.conc_data)

        # If the b_idx is to close to the end of measurement
        if b_idx + post_b_idx > np.shape(conc_data)[3]:
            self.clustIdcs = range(b_idx - pre_b_idx, np.shape(conc_data)[3])

        # Or it's to close to the beginning
        elif b_idx - pre_b_idx < 0:
            self.clustIdcs = range(0, b_idx + post_b_idx)

        else:
            self.clustIdcs = range(b_idx - pre_b_idx, b_idx + post_b_idx)

        self.concData2D = np.reshape(
            conc_data[self.ROI3_4D], (np.sum(self.ROI3), np.shape(self.conc_data)[3])
        )
        self.t4 = np.array(self.clustIdcs)

    def clustering(self):
        if self.clustercount == 1:
            concData2D = np.array(self.concData2D)
            ROI = np.array(self.ROI3)

        if self.clustercount == 2:
            concData2D = np.array(self.concDatalowfM_preClust)
            ROI = np.array(self.ROI_preClust)

        firstM_data = np.array(self.firstM_data)
        clustIdcs = range(self.clustIdcs.start, self.clustIdcs.stop)
        numClust = int(self.numClust)

        # Normalize Curves to have same auc (=1)
        summed = np.sum(concData2D[:, clustIdcs], axis=1)
        summedrep = np.repeat(
            summed[:, np.newaxis], np.shape(concData2D[:, clustIdcs])[1], axis=1
        )
        concData2DNorm = concData2D[:, clustIdcs] / summedrep

        # Divide normlized curves into three groups based on firstMoment
        firstM_sort = np.argsort(firstM_data[ROI])
        firstM_split = np.array_split(
            firstM_data[ROI], numClust
        )  # Lika stora intervall av vektorn concDataNorm

        # Create intervals and starting values (C_in) for the k-means clustering
        intervals = [0]
        for i in range(0, numClust):
            intervals.append(intervals[i] + len(firstM_split[i]))

        C_in = np.zeros((numClust, np.shape(concData2DNorm)[1]))
        for n in range(0, len(intervals) - 1):
            C_in[n, :] = np.mean(
                np.take(
                    concData2DNorm, firstM_sort[intervals[n] : intervals[n + 1]], axis=0
                ),
                axis=0,
            )

        # Clustering
        k_means = KMeans(n_clusters=numClust, init=C_in).fit(concData2DNorm)

        # Sort cluster result based on first moment, the onw with lowest will be the "AIF group"
        clusterOrder = np.sum(self.t4 * k_means.cluster_centers_, axis=1).argsort()
        AIF_group = clusterOrder[0]
        concDatalowfM = concData2D[k_means.labels_ == AIF_group]

        # Create a ROI from the result
        AIF_idcs = np.zeros(np.shape(concData2D)[0])
        for data in concDatalowfM:
            AIF_idcs[np.argwhere((concData2D == data).all(axis=1))[0][0]] = True

        ROI_clust = np.zeros(np.shape(ROI)).astype(bool)
        ROI_clust[ROI] = AIF_idcs

        self.clustercount += 1

        return k_means, concDatalowfM, concData2DNorm, ROI_clust, clusterOrder

    def calc_aif(self):
        if self.do_Mouridsen2006:
            # ROI avarage gives the AIF
            self.AIF = np.mean(self.concDatalowfM_finalClust, axis=0)
        else:
            #
            clustIdcs = range(self.clustIdcs.start, self.clustIdcs.stop)
            sortIdx = np.argsort(
                np.sum(self.t4 * self.concDatalowfM_finalClust[:, clustIdcs], axis=1)
            )  # Sort by first moment

            concDataLowest3 = self.concDatalowfM_finalClust[
                sortIdx[:3]
            ]  # 3 Curves of the concentration data with lowest first moment
            highAUC_idx = np.argmax(
                np.sum(concDataLowest3[:, clustIdcs], axis=1)
            )  # The curve that has the highest AUC
            concDataLowest3Mean = np.mean(concDataLowest3, 0)
            scaleFact = np.sum(concDataLowest3[:, clustIdcs][highAUC_idx]) / np.sum(
                concDataLowest3Mean[clustIdcs]
            )  # Scales the mean value to the curve with highest AUC
            concDataLowest3MeanScaled = (
                concDataLowest3Mean * scaleFact
            )  # This is the final AIF

            # Getting the index where the 3 lowest first moments are and putting them in ROI5.
            firstM_all = np.sum(self.t4 * self.concData2D[:, clustIdcs], axis=1)
            lowest3firstM = np.sum(self.t4 * concDataLowest3[:, clustIdcs], axis=1)
            lowest3 = np.zeros(np.shape(firstM_all))

            for firstMom in lowest3firstM:
                lowest3[np.argwhere(firstMom == firstM_all)[0][0]] = 1

            ROI_AIF = np.zeros(np.shape(self.ROI3))
            ROI_AIF[self.ROI3] = lowest3

            self.ROI_AIF = ROI_AIF
            self.AIF = concDataLowest3MeanScaled

    def return_vals(self):
        return self.AIF, self.ROI_AIF
