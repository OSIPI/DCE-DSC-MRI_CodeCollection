import os
from xmlrpc.client import boolean

import numpy as np
#import pandas as pd
import scipy.io as sio
import nibabel as nib

def sig2conc(S,kappa,te,S0):
    return -np.log(S/S0)/(kappa*te) # S = S0*np.exp(-kappa*C*TE)

def dsc_DRO_data_vascular_model(delay=0):
    """
    Import dsc signal data for testing.
    Convert to concentration domain using provided simulation data S0, te, kappa.

    Data summary: digital reference object consisting of signal time curves 
    representing perfusion scenarios typical of grey and white matter.

    Source: https://github.com/arthur-chakwizira/BezierCurveDeconvolution
    Ref: Non-parametric deconvolution using Bézier curves for quantification 
    of cerebral perfusion in dynamic susceptibility contrast MRI

    Similar digital reference objects have previously been used in publications 
    evaluating novel perfusion deconvolution/estimation techniques including Wu et al. 2003 
    (DOI 10.1002/mrm.10522), Mouridsen et al. 2006 (DOI 10.1016/j.neuroimage.2006.06.015 ), 
    Chappell et al. 2015 (10.1002/mrm.25390)
    
    Parameters
    ----------
    Transit time distribution: gamma variate distribution 
    with shape parameter lambda = 3 
    
    delay : Bool
        Not applied yet

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case

    """
    delay_aif = str(delay) #sim_cfg['delay_aif'][0][0][0]
    # For later: make these input variables:
    delay_level = 'high'
    disperse_aif = '0'
    disp_level = 'high'

    # Read .mat file
    datadir = os.path.join(os.path.dirname(__file__), 'data')
    sim_cfg = sio.loadmat(datadir + '/sim_config' + '_delay_' + delay_aif  + '.mat')['sim_config']
    cbf_g = sim_cfg['cbf_g'][0][0][0].tolist()
    cbv_g = sim_cfg['cbv_gray'][0][0][0][0].tolist()
    cbf_w = sim_cfg['cbf_w'][0][0][0].tolist()
    cbv_w = sim_cfg['cbv_white'][0][0][0][0].tolist()
    snr_t = sim_cfg['snr_t'][0][0][0][0].tolist()
    image_size = sim_cfg['image_size'][0][0][0].tolist()
    n_slices = sim_cfg['n_slices'][0][0][0][0].tolist()
    n_time_points = sim_cfg['n_time_points'][0][0][0][0].tolist()
    te = sim_cfg['te'][0][0][0][0].tolist() # Seconds
    tr = sim_cfg['tr'][0][0][0][0].tolist() # Seconds
    dk = sim_cfg['dk'][0][0][0]
    residue_function = sim_cfg['residue_function'][0][0][0]
    lamda = sim_cfg['lambda'][0][0][0][0].tolist()
    S0 = sim_cfg['S0'][0][0][0][0].tolist()#100
    C_gray = dict()
    C_white = dict()
    kappa_a = sio.loadmat(datadir + '/kappa_a.mat')['kappa_a'][0][0]
    kappa_s = sio.loadmat(datadir + '/kappa_s.mat')['kappa_s'][0][0]

    img = nib.load(datadir + '/aif.nii')
    data = img.get_data() 
    S_aif = np.squeeze(data)
    C_aif = sig2conc(S_aif,kappa_a,te,S0)

    for i, cbf_gray in enumerate(cbf_g):
        label = '/St_disp_' + str(disperse_aif) +  '_' + dk + '_lvl_' + disp_level + '_delay_' + delay_aif + '_lvl_' + delay_level

        if residue_function == 'gamma_dist':
            label = label + '_resfunc_' + residue_function + '_lambda_' + str(lamda) + '_cbf_' + str(cbf_gray)
        elif residue_function == 'linear':
            label = label + '_resfunc_' + residue_function + '_lambda_' + '_cbf_' + str(cbf_gray)
        else:
            label = label + '_cbf_gw_' + str(cbf_gray) + '_cbf_wm_' + str(cbf_w[i])
        
        img = nib.load(datadir + label + '.nii')
        data = img.get_data()
        data = np.squeeze(data) # 4D -> 3D
        S_data2D = data.reshape([data.shape[0]*data.shape[1], data.shape[2]],order =  'C')

        # Convert to concentration 
        C_data2D = sig2conc(S_data2D,kappa_s,te,S0)
        cutRow = int(C_data2D.shape[0]/2-1)
        C_gray[str(cbf_g[i])] = C_data2D[0:cutRow, :]
        C_white[str(cbf_w[i])] = C_data2D[cutRow+1 : C_data2D.shape[0], :]

    # set the tolerance to use for this dataset
    r_tol_cbv_g = [0.1]
    r_tol_cbv_w = [0.1]
    r_tol_cbf_g = [0.1] * len(cbf_g)
    r_tol_cbf_w = [0.1] * len(cbf_w)

    a_tol_cbv_g = [1]
    a_tol_cbv_w = [1]
    a_tol_cbf_g = [15] * len(cbf_g)
    a_tol_cbf_w = [15] * len(cbf_w)

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(
        zip(label, [C_gray], [C_white], [C_aif], [tr], [cbv_g], [cbv_w], [cbf_g], [cbf_w],
            r_tol_cbv_g, r_tol_cbv_w, r_tol_cbf_g, r_tol_cbf_w, 
            a_tol_cbv_g, a_tol_cbv_w, a_tol_cbf_g, a_tol_cbf_w))

    return pars