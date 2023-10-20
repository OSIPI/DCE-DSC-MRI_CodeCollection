#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 01:58:23 2021

@authors: Sudarshan Ragunathan, Laura C Bell
@email: sudarshan.ragunathan@barrowneuro.org
@institution: Barrow Neurological Institute, Phoenix, AZ, USA
@lab: Translational Bioimaging Group (TBG)

DESCRIPTION
-----------
This function performs BSW leakage correction on ∆R2* values and has the following Input(s)/Ouput(s):
    INPUTS:
        dR2s - ∆R2* values [s-1]
        nonenhance_map - Whole Brain Non-Enhancing (WBNE) mask
    OUTPUTS:
        dR2s_leakagecorrected - matrix with leakage corrected ∆T2* values
"""

import numpy as np
from scipy.optimize import curve_fit

def BSWfunction(X,k1,k2):
    
        dR2s_tumor_WBNE,dR2s_WBNEint = X
        return k1*dR2s_tumor_WBNE - k2*dR2s_WBNEint
    
    
def BSWleakagecorr(dR2s, nonenhance_map):
    
    # Check data dimensionality
    if dR2s.ndim == 4:
        nX,nY,nZ,nDyn = dR2s.shape
        nTE = 1
    else:
        print('Error: Image data has incorrect dimensions (Must be 5D for multi-echo and 4D for single-echo')
        return(1)
    
    # Generate WBNE voxels from ∆R2* map
    dR2s_WBNE = dR2s * np.broadcast_to(np.expand_dims(nonenhance_map,axis=-1),dR2s.shape)  #define WBNE
    dR2s_WBNE = np.squeeze(np.reshape(dR2s_WBNE,(nX*nY*nZ,nTE,nDyn),order='F'))
    dR2s_WBNE[np.isinf(dR2s_WBNE)] = 'nan'    # Adding infinite values to list of NaNs that will be omitted during fitting. 
    dR2s_vec = np.squeeze(np.reshape(dR2s,(nX*nY*nZ,nTE,nDyn),order='F'))
    dR2s_WBNE_avg = np.squeeze(np.nanmean(dR2s_WBNE,axis=0))
    dR2s_WBNE_integral = np.cumsum(dR2s_WBNE_avg,axis=0)
    
    K1_dR2s = np.zeros((nX*nY*nZ))
    K2_dR2s = np.zeros((nX*nY*nZ))
    dR2s_BSW = np.zeros((nX*nY*nZ,nDyn))
    
    p0 = 1.,0.1 #initial guess for fit 
    for x in range(nX*nY*nZ):
        print("Voxel "+str(x))
        if np.all(~np.isnan(dR2s_vec[x,:])) and np.all(~np.isinf(dR2s_vec[x,:])):
            BSW_kvals,BSW_fitcov = curve_fit(BSWfunction,(dR2s_WBNE_avg,dR2s_WBNE_integral),dR2s_vec[x,:],p0)
            K1_dR2s[x] = BSW_kvals[0]
            K2_dR2s[x] = BSW_kvals[1]
            leakage = (-K2_dR2s[x]) * dR2s_WBNE_integral
            dR2s_BSW[x,:] = dR2s_vec[x,:] - leakage;
    
    dR2s_leakagecorrected = np.zeros(dR2s.shape)
    dR2s_leakagecorrected = np.reshape(dR2s_BSW,dR2s.shape,order='F')
    return dR2s_leakagecorrected
