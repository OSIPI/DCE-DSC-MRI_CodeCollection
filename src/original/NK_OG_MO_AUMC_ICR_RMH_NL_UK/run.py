"""
July 2024 by Natalia Korobova, Oliver Gurney-Champion and Matthew Orton
n.korobova@amsterdamumc.nl
o.j.gurney-champion@amsterdamumc.nl

Contains the modification of Extended Tofts model for MRI acquisitions oversampling the center of k-space, such as radial, spiral, PROPELLER trajectories.
Modification assumes that the image contrast is averaged over an acquisition time of one dynamic as a result of oversampling the center of k-space.
Modification is based on previously published code by OG_MO_AUMC_ICR_RMH_NL_UK from the OSIPI GitHub.

requirements:
scipy
joblib
matplotlib
numpy
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import DCE_correction as dce
import utils

Nsamples = 20 # number of samples (simulated voxels) 
num_images = 38 # number of acquired images - defines temporal resoltion!
total_time = 5 # total time of dyncamic acquisition in minutes
''' temporal resolution in seconds = total_time // (num_images-1) * 60 
                        301 images: 1 sec/frame
                        101 images: 3 sec/frame
                        76 images:  4 sec/frame
                        61 images:  5 sec/frame
                        38 images:  7.933333 sec /frame
                        31 images:  10 sec/frame
                        21 images:  15 sec/frame
                        26 images:  12 sec/frame
                        11 images:  30 sec/frames
'''
aif_fit = True # True when the AIF model parameters should be calculated based on a given AIF curve
aif_generate = True # True when different AIF curves should be generated per voxel
SNR = 30 # SNR level. If SNR = -1 no noise added
vp_uniform = False # True when vp should be distributed uniformly; False when vp should follow an exponential distribution
jobs = 1 # number of threads used to fit a PK model

GT_init = True # True

def main():

    continuous_time = np.linspace(0, total_time, 3001)              # approximation of a continuous time series
    realistic_time = np.linspace(0, total_time, num_images)         # sampling time series
    ke,dt,ve,vp = utils.generate_parameters(Nsamples,vp_uniform)    # initialize DCE parameters
    delay = realistic_time[1]                                       # acquisiotion time of one dynamic image
    if aif_generate:
        aif = utils.generate_aif(Nsamples)
    else:
        aif = [dce.aifPopPMB()] * Nsamples

    if aif_fit:                                                                      # parametrizing AIF curve to an anlytical AIF model proposed by Matthew Orton
        true_AIF = utils.calc_aif(continuous_time, aif, delay=0, jobs=jobs)          # ground truth AIF (calculated at very small time steps)
        Cartesian_AIF_measured = utils.calc_aif(realistic_time - delay / 2, aif, delay=0, jobs=jobs)     # AIF measued Cartesionally - k-space sampled once -> image contrast captured instantaneously
        radial_AIF_measured = utils.calc_aif(realistic_time, aif, delay=delay, jobs=jobs)                # AIF measued radially or spirally - k-space oversampled within one dynamic image -> image contrast formed over the acqisition time

        if GT_init:
            ab = [x['ab'] for x in aif]
            ae = [x['ae'] for x in aif]
            mb = [x['mb'] for x in aif]
            me = [x['me'] for x in aif]
            t0 = [x['t0'] for x in aif]
            X0 = np.stack((ab, ae, mb, me, t0), axis=1)
        else: X0 = (9,1.5,23,0.1,0.5)

        # Fit AIF curve to AIF model
        Cartesian_AIF_params = dce.fit_aif(realistic_time - delay / 2, Cartesian_AIF_measured, delay=0, jobs = jobs, X0=X0)
        radial_AIF_fit_conventional_params = dce.fit_aif(realistic_time - delay / 2, radial_AIF_measured, delay=0, jobs = jobs, X0=X0)
        radial_AIF_fit_corrected_params = dce.fit_aif(realistic_time, radial_AIF_measured, delay=delay, jobs = jobs, X0=X0)

        # Calculate AIF curve based on estimated AIF parameters
        Cartesian_AIF_curve = utils.calc_aif(continuous_time - delay / 2, Cartesian_AIF_params, delay=0, jobs=jobs)
        radial_AIF_fit_conventional_curve = utils.calc_aif(continuous_time - delay / 2,radial_AIF_fit_conventional_params, delay=0, jobs=jobs)
        radial_AIF_fit_corrected_curve = utils.calc_aif(continuous_time - delay / 2, radial_AIF_fit_corrected_params, delay=0, jobs = jobs)


        for aa in range(min(3,Nsamples)):
            plt.figure()
            plt.title(f'Sample {aa}: AIF')
            plt.plot(continuous_time, true_AIF[aa], color='blue', linestyle='-', label='Ground truth AIF')
            plt.plot(realistic_time-delay/2 , Cartesian_AIF_measured[aa], 'ok', markersize=6, label='Cartesian sampling')
            plt.plot(realistic_time-delay/2, radial_AIF_measured[aa], '*r', markersize=6, label='Radial sampling')
            plt.plot(continuous_time-delay/2, Cartesian_AIF_curve[aa], 'green', linestyle='--', label='Conventional fit to Catesian sampling')
            plt.plot(continuous_time-delay/2, radial_AIF_fit_conventional_curve[aa], 'magenta', linestyle='--', label='Conventional fit to radial sampling')
            plt.plot(continuous_time-delay/2, radial_AIF_fit_corrected_curve[aa], 'cyan', linestyle='--',label='Corrected fit to Catesian sampling')
            plt.legend()
            plt.show()
    else:                                                                            # assume we know the correct AIF
        Cartesian_AIF_params = aif
        radial_AIF_fit_corrected_params = aif
        radial_AIF_fit_conventional_params = aif

    true_C = utils.calc_concentration(continuous_time, aif, ke, dt, ve, vp, delay=0,jobs=jobs,snr=-1)                        # ground truth concentration curve (calculated at very small time steps)
    Cartesian_measured_C = utils.calc_concentration(realistic_time-delay/2, aif, ke, dt, ve, vp, delay=0, jobs=jobs,snr=SNR) # curve measued Cartesionally - k-space sampled once -> image contrast captured instantaneously
    radial_measured_C = utils.calc_concentration(realistic_time, aif, ke, dt, ve, vp, delay=delay, jobs=jobs,snr=SNR)        # curve measued radially or spirally - k-space oversampled within one dynamic image -> image contrast formed over the acqisition time


    if GT_init: X0 = np.stack((ke,dt,ve,vp),axis=1)
    else: X0=(1.5, 1.0, 0.5, 0.5)

    # Fit concentration curve to extended Tofts
    # for radial / spiral acquisition both, conventional and corrected Estended Tofts models, are applied
    ke_Cartesian_fit, dt_Cartesian_fit, ve_Cartesian_fit, vp_Cartesian_fit = dce.fit_tofts_model(realistic_time - delay / 2, Cartesian_measured_C, delay= 0, aif = Cartesian_AIF_params, jobs = jobs, X0=X0)
    ke_radial_conventional_fit, dt_radial_conventional_fit, ve_radial_conventional_fit, vp_radial_conventional_fit = dce.fit_tofts_model(realistic_time - delay / 2, radial_measured_C, delay = 0, aif = radial_AIF_fit_conventional_params, jobs = jobs, X0=X0)
    ke_radial_corrected_fit, dt_radial_corrected_fit, ve_radial_corrected_fit, vp_radial_corrected_fit = dce.fit_tofts_model(realistic_time, radial_measured_C, delay=delay, aif=radial_AIF_fit_corrected_params, jobs=jobs,X0=X0)

    # Calculate concentration curves based on estimated Pk parameters
    C_Cartesian_fit = utils.calc_concentration(continuous_time - delay / 2, Cartesian_AIF_params, ke_Cartesian_fit,
                                               dt_Cartesian_fit, ve_Cartesian_fit, vp_Cartesian_fit, delay=0, jobs=jobs, snr=-1)
    C_radial_fit_conventional = utils.calc_concentration(continuous_time - delay / 2, radial_AIF_fit_conventional_params,
                                                   ke_radial_conventional_fit, dt_radial_conventional_fit,
                                                   ve_radial_conventional_fit, vp_radial_conventional_fit, delay = 0,
                                                   jobs=jobs, snr=-1)
    C_radial_fit_corrected = utils.calc_concentration(continuous_time - delay / 2, radial_AIF_fit_corrected_params,
                                                ke_radial_corrected_fit, dt_radial_corrected_fit,
                                                ve_radial_corrected_fit, vp_radial_corrected_fit, delay = 0, jobs=jobs, snr=-1)
    for aa in range(min(3, Nsamples)):
        plt.figure()
        plt.title(f'Sample {aa}: Concentration curve')
        plt.plot(continuous_time, true_C[aa], color='blue', linestyle='-',linewidth=3.0, label='Ground truth contrast')
        plt.plot(realistic_time-delay/2, Cartesian_measured_C[aa], 'ok', markersize=6,  label='Cartesian sampling')
        plt.plot(realistic_time-delay/2, radial_measured_C[aa], '*r', markersize=6, label='Radial sampling')
        plt.plot(continuous_time - delay / 2, C_Cartesian_fit[aa], 'green', linestyle='--', label='Conventional fit to Catesian sampling')
        plt.plot(continuous_time - delay / 2, C_radial_fit_conventional[aa], 'magenta', linestyle='--', label='Conventional fit to radial sampling')
        plt.plot(continuous_time - delay / 2, C_radial_fit_corrected[aa], 'cyan', linestyle='--', label='Corrected fit to radial sampling')
        plt.legend()
        plt.show()

    for parameter, gt, Cartesian, radial_conventional, radial_corrected in zip(['vp','ve','ke','dt'],
                                                                              [vp, ve, ke, dt],
                                                                              [vp_Cartesian_fit,ve_Cartesian_fit,ke_Cartesian_fit,dt_Cartesian_fit],
                                                                              [vp_radial_conventional_fit,ve_radial_conventional_fit,ke_radial_conventional_fit,dt_radial_conventional_fit],
                                                                              [vp_radial_corrected_fit,ve_radial_corrected_fit,ke_radial_corrected_fit,dt_radial_corrected_fit]):
        error_cartesian = Cartesian - gt
        error_radial_conv = radial_conventional - gt
        error_radial_corr = radial_corrected - gt

        randerror_cartesian = np.std(error_cartesian)
        randerror_radial_conv = np.std(error_radial_conv)
        randerror_radial_corr = np.std(error_radial_corr)

        syserror_cartesian = np.mean(error_cartesian)
        syserror_radial_conv = np.mean(error_radial_conv)
        syserror_radial_corr = np.mean(error_radial_corr)
        print(f'-------------------Parameter {parameter}---------------------')
        print(f'Cartesian:\nrandom error {randerror_cartesian:.4f}\tsystematic error {syserror_cartesian:.4f}')
        print(f'Radial conventional:\nrandom error {randerror_radial_conv:.4f}\tsystematic error {syserror_radial_conv:.4f}')
        print(f'Radal corrected:\nrandom error {randerror_radial_corr:.4f}\tsystematic error {syserror_radial_corr:.4f}')

if __name__ == "__main__":
    main()