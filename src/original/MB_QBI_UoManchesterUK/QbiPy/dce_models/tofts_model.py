'''
Module for woring with the extended-Tofts model.

We provide the forward model (concentration_from_model), to be used elsewhere in
fitting if required. In addition we provide methods for fitting the model using
a linear-least squares approach.

The model includes the standard 3 parameters Ktrans, ve, vp. It also includes a delay
parameter tau_a, which is used to interpolate/delay the AIF to given time.

All times are assumed to be in minutes.

The AIF must be a QbiPy AIF object (see dce_aif). However if you have a set of AIF values (Ca_t)
and associated dynamic times (t), it is trivial to create an AIF object:

aif = dce_aif.Aif(times = t, base_aif=Ca_t, aif_type=ARRAY)

The remaining model parameters can either be input as scalars, or 1D np.arrays. The two forms
can be mixed, but any paramaters set as arrays must be the same length. The output is always
a 2D array C(t) = (n_samples x n_times).

The main concentration_from_model function is written this way because it is primarily used
for setting up ground truth inputs from Monte-Carlo simulations. However, for convenience
if using as a forward model during model fits, a wrapper function is provided in which
a single set of model parameters are input as a list/tuple/array and C(t) is returned
as a 1D-array

'''

import numpy as np
from QbiPy.dce_models import dce_aif 
from QbiPy import helpers

#
#---------------------------------------------------------------------------------
def concentration_from_model(aif:dce_aif.Aif, 
    Ktrans: np.array, v_e: np.array, v_p: np.array, tau_a: np.array, 
    use_exp_conv:bool=False, all_scalar=False)->np.array:
    '''
    Compute concentration time-series of extended-Tofts model from input
    paramaters
    
     Inputs:
        aif (Aif object, num_times): 
            object to store and resample arterial input function values
    
        Ktrans (1D np.array, num_voxels): 
            Ktrans values, 1 for each voxel or scalar
    
        v_p (1D np.array, num_voxels): 
            vascular volume fraction values, 1 for each voxel or scalar
    
        v_e (1D np.array, num_voxels): 
            extra-cellular, extra-vascular volume fraction values, 1 for each voxel or scalar
    
        tau_a (1D np.array, num_voxels): 
            arterial delay values, 1 for each voxel or scalar
    
        use_exp_conv, bool: 
            if true, uses non-interpolating exponential convolution, otherwise does standard stepwise

        all_scalar, bool: 
            if true, skips checks on parameter dimensions, and runs for a single voxel
    
     Outputs:
        C_model (2D np.array, num_voxels x num_times):
            Model concentrations at each time point for each voxel computed from model paramaters
    '''

    if all_scalar:
        num_voxels = 1
    else:
        #We allow the model paramaters to be scalar, whilst also accepting higher dimension arrays
        num_voxels,Ktrans, v_e, v_p, tau_a = helpers.check_param_size(
            Ktrans=Ktrans,v_e=v_e,v_p=v_p, tau_a=tau_a
        )

    #precompute exponential
    k_ep = Ktrans / v_e

    #Make time relative to first scan, and compute time intervals
    num_times = aif.times_.size
    t = aif.times_

    #create container for running integral sum
    #integral_sum = np.zeros(num_voxels) #1d nv

    #Resample the AIF
    aif_offset = aif.resample_AIF(tau_a) #nv x nt
    
    #Create container for model concentrations
    C_model = np.zeros([num_voxels, num_times])

    e_i = 0
    for i_t in range(1, num_times):
        
        #Get current time, and time change
        t1 = t[i_t] #scalar
        delta_t = t1 - t[i_t-1] #scalar
        
        #Compute (tau_a) combined arterial and vascular input for this time
        Ca_t0 = aif_offset[:,i_t-1]#1d n_v
        Ca_t1 = aif_offset[:,i_t]#1d n_v

        if use_exp_conv:
            e_i = helpers.exp_conv(k_ep, delta_t, Ca_t1, Ca_t0, e_i)
        
        else:
            #Update the exponentials for the transfer terms in the two compartments
            e_delta = np.exp(-delta_t * k_ep) #1d n_v         
            
            A = delta_t * 0.5 * (Ca_t1 + Ca_t0*e_delta)
            e_i = e_i * e_delta + A

        #Combine the two compartments with the rate constant to get the final
        #concentration at this time point
        C_model[:,i_t] = v_p * Ca_t1 + Ktrans * e_i

    return C_model

#
#---------------------------------------------------------------------------------
def concentration_from_model_single(params: np.array, aif:dce_aif.Aif)->np.array:
    '''
    Compute concentration time-series of extended-Tofts model from input
    paramaters
    
     Inputs:
        params (tuple/list/1D np.array): 
            4 element array containing [Ktrans, v_e, v_p, tau_a] for a single sample
        
        aif (Aif object, num_times): 
            object to store and resample arterial input function values
    
     Outputs:
       C_model (1D np.array num_times) - Model concentrations at each time point for each 
       voxel computed from model paramaters
    '''
    return concentration_from_model(aif, 
        params[0], params[1], params[2], params[3], 
        use_exp_conv=False, all_scalar=True)[0,]

#
#---------------------------------------------------------------------------
def construct_LLS_matrix(Ctis_t:np.array, aif:dce_aif.Aif, tau_a:float):
    '''
    Make a matrix for linear least-sqaures (LLS) solving for a single tissue time-series 
	
	Inputs:
        Ct_sig: np.array (num_times)
            time-series of signal derived CA concentration

        aif (Aif object, num_times): 
            object to store and resample arterial input function values (1 for each time point)
    
        tau_a: float 
            arterial delay values, 1 for each voxel

    Outputs:
        A_:np.array (num_times x 3)
            Matrix for LLS solver collapsed column major to a single data vector

    '''
    t = aif.times_
    Cp_t = aif.resample_AIF(tau_a)[0,]
    n_t = aif.num_times()
    
    A_ = np.zeros((n_t,3))

    Cp_t_int = helpers.trapz_integral(Cp_t, t)
    Ctis_t_int = helpers.trapz_integral(Ctis_t, t)

    A_[:,0] = Cp_t_int
    A_[:,1] = -Ctis_t_int
    A_[:,2] = Cp_t

    return A_

#
#---------------------------------------------------------------------------
def solve_LLS(Ctis_t:np.array, aif:dce_aif.Aif, tau_a:float):
    '''
    Solve model parameters for a single tissue time-series using LLS
	
	Inputs:
        Ct_sig: np.array (num_times)
            time-series of signal derived CA concentration

        aif (Aif object, num_times): 
            object to store and resample arterial input function values (1 for each time point)
    
        tau_a: float 
            arterial delay values, 1 for each voxel

    Outputs:
        Ktrans, v_e, v_p : float
            TK model parameters
    '''
    A_ = construct_LLS_matrix(Ctis_t, aif, tau_a)
    C_ = Ctis_t
    B_ = np.linalg.lstsq(A_, C_, rcond=None)[0]
    k_2 = B_[1]
    v_p = B_[2]
    Ktrans = B_[0] - k_2*v_p
    v_e = Ktrans / k_2
    return Ktrans, v_e, v_p