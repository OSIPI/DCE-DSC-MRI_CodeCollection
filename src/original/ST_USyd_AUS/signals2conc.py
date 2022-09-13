#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 20:33:11 2019

@author: sirishatadimalla
"""

import numpy as np
from scipy.optimize import fsolve
from joblib import Parallel, delayed

def spgr_func(x, *spgr_params):
    r1, FA, TR, R10, S0, S = spgr_params
    E0 = np.exp(-TR*R10)
    E1 = np.exp(-TR*r1*x)*E0
    c = np.cos(FA*np.pi/180)
    out = S - S0*(1-E1)*(1-c*E0)/(1-E0)/(1-c*E1)
    return(out)
    
def signals2conc(time, S, FA, TR, precontrastR1, relaxivity, no_of_baseline_scans):
    S_baseline = np.mean(S[0:no_of_baseline_scans])
    conc_initial = np.zeros(len(time))
    conc = [Parallel(n_jobs=4)(delayed(fsolve)(spgr_func, x0=conc_initial[p], args = (relaxivity, FA, TR, precontrastR1, S_baseline, S[p])) for p in np.arange(0,len(time)))]
    conc = np.squeeze(np.array(conc))
    return(conc)
            
            