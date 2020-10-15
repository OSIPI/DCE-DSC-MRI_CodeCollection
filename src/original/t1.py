# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:50:09 2020.

@author: Michael Thrippleton

Functions to fit MRI signal to obtain T1:
    fit_vfa_2_point: obtain T1 using analytical formula based on two images
    fit_vfa_linear: obtain T1 using linear regression
    fit_vfa_nonlinear: obtain T1 using non-linear least squares fit

"""
from functools import wraps

import numpy as np
from scipy.optimize import curve_fit

from . import signal

def apply_to_image(f):    
    """Decorate fitting functions to apply to multiple voxels/regions.
    
    Also applies a mask to avoid calculating invalid voxels.
    
    Parameters
    ----------
    f : Function that operates on single vector of numbers and returns a single
        set of parameters. First argument should be a 1-D np array.

    Returns
    -------
    wrapper: function that takes a (N+1)-D signal input and applies f to every
        voxel by iterating over the N spatial dimensions and returning parameters
        as N-D np arrays

    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        s = args[0]
        s_2d = s.reshape(-1,s.shape[-1]) #convert (N+1)-D array to matrix of 1-D signal vectors
        if 'mask' in kwargs:
            mask_1d = np.atleast_1d(kwargs['mask'].reshape([-1])) #reshape mask            
        else:
            mask_1d = np.ones(s_2d.shape[0])

        #apply f to each 1-D signal vector
        s0, t1_s = zip(*[f(s_1d,*args[1:]) if mask_1d[idx] != 0 else (np.nan, np.nan) for idx, s_1d in enumerate(s_2d)])
        
        #reshape output to N spatial dimensions
        return np.asarray(s0).reshape(s.shape[:-1]), np.asarray(t1_s).reshape(s.shape[:-1])
    return wrapper


def fit_vfa_2_point(s, fa_rad, tr_s, idx = (0,-1), mask=None):
    """Return T1 based on 2 SPGR signals with different FA but same TR and TE.
    
    Parameters
    ----------
        s: np array of signals. Can be >=1 dimensional but last dimension indicates
            which acquisition (i.e. flip angle) is referred to.
        fa_rad: 1-D np array of flip angles
        tr_s: TR
        idx, optional: tuple containing two indices, indicating acquisitions to use for
            two-point T1 estimation. Default is to use first and last.
        mask, optional: np array corresponding to mask image. Nans are returned
            where mask value is 0. Default is to process all voxels.                                                              
       
    Returns
    -------
        t1_s: T1
        s0: signal corresponding to equilibrium magnetisation excited with 90
            degree pulse
            
    """       
    s1, s2 = np.atleast_1d(s[..., idx[0]]), np.atleast_1d(s[..., idx[1]])    
    fa1_rad, fa2_rad = fa_rad[idx[0]], fa_rad[idx[1]]
    with np.errstate(divide='ignore', invalid='ignore'):
        sr=s1/s2
        t1_s = tr_s / np.log( (sr*np.sin(fa2_rad)*np.cos(fa1_rad) - np.sin(fa1_rad)*np.cos(fa2_rad)) / (sr*np.sin(fa2_rad) - np.sin(fa1_rad)) )
        s0 = s1 * ( (1-np.exp(-tr_s/t1_s)*np.cos(fa1_rad)) / ((1-np.exp(-tr_s/t1_s))*np.sin(fa1_rad)) )
    
    t1_s[~np.isreal(t1_s) | (t1_s<=0) | np.isinf(t1_s)] = np.nan
    s0[(s0<=0) | np.isinf(s0)] = np.nan    
    
    if mask is None:
        mask = np.ones(s1.shape)        
    
    t1_s[mask==0] = np.nan
    s0[mask==0] = np.nan
    
    return s0, t1_s


@apply_to_image
def fit_vfa_linear(s, fa_rad, tr_s):
    """Return T1 based on VFA signals using linear regression.
    
    (decorated by apply_to_image)
    
    Parameters
    ----------
        s: np array of signals. Can be >=1 dimensional but last dimensions should
            correspond to acquisition with particular flip angle
        fa_rad: 1-D np array of flip angles
        tr_s: TR
        mask, optional: np array corresponding to mask image. Nans are returned
            where mask value is 0. Default is to process all voxels.  

    Returns
    -------
        t1_s: T1
        s0: signal corresponding to equilibrium magnetisation excited with 90
            degree pulse            
    """
    y = s / np.sin(fa_rad)
    x = s / np.tan(fa_rad)
    A = np.stack([x, np.ones(x.shape)], axis=1)
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    is_valid = (intercept > 0) and (0. < slope < 1.)
    t1_s, s0 = (-tr_s/np.log(slope), intercept/(1-slope)) if is_valid else (np.nan, np.nan)
    
    return s0, t1_s


@apply_to_image
def fit_vfa_nonlinear(s,fa_rad,tr_s):
    """Return T1 based on VFA signals using NLLS fitting.
    
    (decorated by apply_to_image)
    
    Parameters
    ----------
        s: np array of signals. Can be >=1 dimensional but last dimensions should
            correspond to acquisition with particular flip angle
        fa_rad: 1-D np array of flip angles
        tr_s: TR
        mask, optional: np array corresponding to mask image. Nans are returned
            where mask value is 0. Default is to process all voxels.  

    Returns
    -------
        t1_s: T1
        s0: signal corresponding to equilibrium magnetisation excited with 90
            degree pulse  
    """
    p_linear = np.array(fit_vfa_linear(s, fa_rad, tr_s)) #linear fit to obtain initial guess
    
    p0 = p_linear if (~np.isnan(p_linear[0]) & ~np.isnan(p_linear[1])) else np.array([10.0*s[0], 1.0])
 
    popt, pcov = curve_fit(lambda x_fa_rad, p_S0, p_T1_s: signal.spgr(p_S0, p_T1_s, tr_s, x_fa_rad), fa_rad, s,
                           p0=p0, sigma=None, absolute_sigma=False, check_finite=True, bounds=(1e-5, np.inf), method='trf', jac=None) 
    s0 = popt[0]
    t1_s = popt[1]
    return s0, t1_s
        



