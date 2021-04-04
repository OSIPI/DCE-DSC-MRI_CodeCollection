#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:33:17 2020

@author: sirishatadimalla
"""
#### This function obtains T1 by fitting VFA signal data to the SPGR equation. Nonlinear or 
#### linearised (faster) SPGR equation may be used. Fitting is performed without any
#### parameter constraints/bounds.
#### ---------------------------------------------------
#### Inputs
#### FA: data array of FA used in the SPGR experiment
#### S: data array of T1w signals obtained at each FA
#### TR: Repetition time in ms
#### method: 'linear' (default) or 'nonlinear'
#### Outputs
#### S0: signal at equilibrium magnetisation
#### T1: calculated T1 in ms
#### Example
#### S0, T1 = VFAT1mapping(FA, S, TR, method = 'linear')
#### ----------------------------------------------------
#
#  Import required packages
import numpy as np
from lmfit import Model

# Nonlinear SPGR equation definition
def spgr_nonlinear(X, S0, TR, T1):
    s = np.sin(X*np.pi/180)
    c = np.cos(X*np.pi/180)
    E = np.exp(-TR/T1)
    Y = S0*s*(1-E)/(1-c*E)
    return(Y)

# Linear equation definition
def spgr_linear(X, A, B):
    Y = A*X + B
    return(Y)

def VFAT1mapping(FA, S, TR, method = 'linear'):

    if method == 'nonlinear':
        # Generate model
        spgr_model = Model(spgr_nonlinear,nan_policy='omit')
        pars = spgr_model.make_params()
        # Initialise fit parameters
        start_T1 = 1000
        start_S0 = np.max(S)
        # Setup fit parameters
        pars.add('TR',value=TR,vary=False)
        pars.add('T1',value=start_T1)
        pars.add('S0',value=start_S0)            
        # Perform fit
        result = spgr_model.fit(S, X=FA, params=pars)
        # Get parameters
        S0 = result.params['S0'].value
        T1 = result.params['T1'].value
        
    else:
        # Generate model
        spgr_model = Model(spgr_linear,nan_policy = 'omit')
        pars = spgr_model.make_params()
        # Get inputs
        X = S/np.tan(FA*np.pi/180)
        Y = S/np.sin(FA*np.pi/180)
        # Initialise fit parameters
        start_A = 0.995
        start_B = 0.005
        # Perform fit
        pars.add('A',value=start_A)
        pars.add('B',value=start_B)            
        # Perform fit
        result = spgr_model.fit(Y, X=X, params=pars)
        # Get parameters
        A = result.params['A'].value
        B = result.params['B'].value
        T1 = -TR/np.log(A)
        S0 = B/(1-A)
    
    return(S0, T1)
