# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:50:09 2020.

@author: Michael Thrippleton

Functions to calculate MRI signals:
    spgr: spoiled gradient echo signal

"""

import numpy as np


def spgr(s0,t1_S,tr_s,fa_rad):      
    """Return signal for SPGR sequence.

    Parameters 
    ----------
        s0 (float or float array): equilibrium signal
        t1_S (float or float array): T1 value
        tr_s (float or float array): TR value
        fa_rad (float or float array): flip angle

    Returns
    -------
        float or float array: steady-state signal

    """
    s=s0 * (((1.0-np.exp(-tr_s/t1_S))*np.sin(fa_rad)) / (1.0-np.exp(-tr_s/t1_S)*np.cos(fa_rad)) );
    
    return s
