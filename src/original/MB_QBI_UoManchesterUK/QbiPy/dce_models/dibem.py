'''
Functions for working with generic dual-input bi-exponential models (DIBEM)

This general function form can be used by the two-compartment exchange model
(2CXM), two compartment filtration module and the active-uptake and efflux model (AUEM).

In each case the specific model needs to define how to convert its physiological
parameters into the 4 parameters F+, F_, K+, K_ of a model IRF:

I(t) = F+ . exp(-t . K+) + F_ . exp(-t . K_)

This model IRF is then convolved with a vascular input function Cp(t), where
Cp(t) can either be single supply, typically assumed to be arterial only,
Cp(t) = Ca(t). Or as a mix of two supplies, typically assumed to be
arterial and venous (eg for the liver, supplied by the hepatic portal vein)
Cp(t) = fa.Ca(t) + (1- fa).Cv(t) where fa is the arterial fraction (0 <= fa <= 1).
'''

import numpy as np
from QbiPy.dce_models import dce_aif
from QbiPy import helpers

#
#-------------------------------------------------------------------------------
def concentration_from_model(aif:dce_aif.Aif, 
    F_pos: np.array, F_neg: np.array, K_pos: np.array, K_neg: np.array,  
    f_a: np.array, tau_a: np.array, tau_v: np.array)->np.array:
    '''
    Compute concentration time-series from model parameters
    Inputs:
        aif (Aif object): 
            object to store and resample arterial input function values (1 for each time point)
    
        F_pos, F_neg, K_pos, K_neg: np.array (1D n_samples)
            bi-exponetial IRF parameters

        f_a: np.array (1D n_samples)
            Arterial mixing fraction, final plasma input is Cp(t) = f_a*Ca(t) + (1-f_a)*Cv(t)

        tau_a: np.array (1D n_samples)
            offset times of arrival for conccentraion for Ca_t

        tau_v: np.array (1D n_samples)
            offset times of arrival for conccentraion for Cv_t
    
     Outputs:
       C_model (2D numpy array, n_samples x n_t) - Model concentrations at each time point for each 
       voxel computed from model paramaters
    
     '''
    K_max = 1e9

    #We allow the model paramaters to be scalar, whilst also accepting higher dimension arrays
    n_vox,F_pos,F_neg,K_pos,K_neg,f_a,tau_a,tau_v = helpers.check_param_size(
        F_pos=F_pos,F_neg=F_neg,K_pos=K_pos,K_neg=K_neg,
        f_a=f_a,tau_a=tau_a,tau_v=tau_v
    )

    #Get AIF and PIF, labelled in model equation as Ca_t and Cv_t
    #Resample AIF and get AIF times
    #Make time relative to first scan, and compute time intervals
    n_t = aif.times_.size
    t = aif.times_

    #Resample the AIF
    Ca_t = aif.resample_AIF(tau_a) #nv x nt

    resample_AIF = np.any(f_a)
    if resample_AIF:
        Ca_t = aif.resample_AIF(tau_a)
    else:
        Ca_t = np.zeros((n_vox,n_t))

    f_v = 1 - f_a
    if np.any(f_v):
        Cv_t = aif.resample_PIF(tau_v, ~resample_AIF, True)
    else:
        Cv_t = np.zeros((n_vox,n_t))

    #Irf is of form: I(t) = F_pos.exp(-tK_pos) + F_neg.exp(-tK_neg)
    #C(t) = I(t) ** Ca(t)
    C_t = np.zeros((n_vox,n_t)) 
    Ft_pos = 0
    Ft_neg = 0

    Cp_t0 = f_a*Ca_t[:,0] + f_v * Cv_t[:,0]

    for i_t in range(1, n_t):

        #Compute combined arterial and vascular input for this time
        Cp_t1 = f_a*Ca_t[:,i_t] + f_v * Cv_t[:,i_t] #n_v,1

        delta_t = t[i_t] - t[i_t-1]

        Ft_pos = helpers.exp_conv(K_pos, delta_t, Cp_t1, Cp_t0, Ft_pos)
        Ft_neg = helpers.exp_conv(K_neg, delta_t, Cp_t1, Cp_t0, Ft_neg)

        #Combine the two exponentials in the final concentration
        C = F_neg * Ft_neg / K_neg + F_pos * Ft_pos / K_pos
        C[np.isnan(C)] = 0

        C_t[:,i_t] = C
        Cp_t0 = Cp_t1
    return C_t
    
    #
#---------------------------------------------------------------------------
def construct_LLS_matrix(Ctis_t:np.array, aif:dce_aif.Aif, f_a:float, tau_a:float, tau_v:float):
    '''
    Make a matrix for linear least-sqaures (LLS) solving for a single tissue time-series 
	
	Inputs:
        Ct_sig: np.array (num_times)
            time-series of signal derived CA concentration

        aif (Aif object): 
            object to store and resample arterial input function values (1 for each time point)
    
        f_a: float
            Arterial mixing fraction, final plasma input is Cp(t) = f_a*Ca(t) + (1-f_a)*Cv(t)
            
        tau_a: float
            offset times of arrival for conccentraion for Ca_t

        tau_v: float
            offset times of arrival for conccentraion for Cv_t

    Outputs:
        A_:np.array (num_times x 3)
            Matrix for LLS solver collapsed column major to a single data vector

    '''
    t = aif.times_
    f_v = 1.0 - f_a

    if not f_v:
        Cp_t = aif.resample_AIF(tau_a)[0,]

    elif not f_a:
        Cp_t = aif.resample_PIF(tau_v, True, True)[0,]

    else:
        Ca_t = aif.resample_AIF(tau_a)[0,]
        Cv_t = aif.resample_PIF(tau_v, False, True)[0,]
        Cp_t = f_a*Ca_t + (1 - f_a)*Cv_t

    n_t = aif.num_times()
    
    A_ = np.zeros((n_t,4))

    Cp_t_int = helpers.trapz_integral(Cp_t, t)
    Cp_t_int2 = helpers.trapz_integral(Cp_t_int, t)
    Ctis_t_int = helpers.trapz_integral(Ctis_t, t)
    Ctis_t_int2 = helpers.trapz_integral(Ctis_t_int, t)

    A_[:,0] = -Ctis_t_int2
    A_[:,1] = -Ctis_t_int
    A_[:,2] = Cp_t_int2
    A_[:,3] = Cp_t_int

    return A_
