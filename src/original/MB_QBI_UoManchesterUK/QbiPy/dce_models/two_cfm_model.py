'''
Native python code for working with the two-compartment filtration model (2CFM). 
This has a bi-exponential IRF, and uses the dibem module to compute a forward model.

The model includes the standard 4 parameters Fp, PS, ve, vp. It also includes a delay
parameter tau_a, which is used to interpolate/delay the AIF to given time.

All times are assumed to be in minutes.

The AIF must be a QbiPy AIF object (see dce_aif). However if you have a set of AIF values (Ca_t)
and associated dynamic times (t), it is trivial to create an AIF object:

aif = dce_aif.Aif(times = t, base_aif=Ca_t, aif_type=ARRAY)

The remaining model parameters can either be input as scalars, or 1D numpy arrays. The two forms
can be mixed, but any paramaters set as arrays must be the same length.

Code for converting 2CXM parameters to DIBEM form is defined below.

---------------------- 2CFM conversions ----------------------------------
 
 See papers: 
    Susmita Basak, David L. Buckley, et al.
    "Analytical validation of single-kidney glomerular filtration rate and split renal 
    function as measured with magnetic resonance renography",
    Magnetic Resonance Imaging,
    Volume 59,
    2019,
    Pages 53-60,
    ISSN 0730-725X,
    https://doi.org/10.1016/j.mri.2019.03.005.

    Flouri, D., Lesnic, D. and Sourbron, S.P. (2016), 
    "Fitting the two-compartment model in DCE-MRI by linear inversion." 
    Magn. Reson. Med., 76: 998-1006. https://doi.org/10.1002/mrm.25991

 See also the public GitHub repo:
    https://github.com/plaresmedima/Basak_et_al_2018/blob/master/function_fitting.py
   
'''

import numpy as np
from QbiPy.dce_models import dce_aif, dibem
from QbiPy import helpers

#
#-------------------------------------------------------------------------------
def params_to_DIBEM(F_p, PS, v_e, v_p, Fp_form = False):
    '''
    compute the derived parameters for the 2CXM
    model given input physiological parameters
    [K_pos, K_neg, F_pos, F_neg] = two_cx_params_phys_to_model(F_p, PS, v_e, v_p)

    Inputs:
        F_p: np.array (1D n_samples) 
          flow plasma rate

        PS: np.array (1D n_samples) 
          extraction flow

        v_e: np.array (1D n_samples) 
          extravascular, extracellular volume

        v_p: np.array (1D n_samples) 
          plasma volume

    Outputs:
        F_pos, F_neg - multipliers in model IRF

        K_pos, K_neg - exponents in model IRF

    Notes:

    This needs updating to deal with the margin cases (eg divide by zero if Fp = 0)
    '''
    dims_sz, F_p, PS, v_e, v_p = helpers.check_param_shape(
        F_p=F_p, PS=PS, v_e=v_e, v_p=v_p
    )

    #TODO - what if F_p or PS zero?
    T_P = v_p/F_p
    T_E = v_e/PS
    T_T = (v_p+v_e)/F_p
    T_pos = T_E
    T_neg = T_P
    E_pos = (T_T-T_neg)/(T_pos-T_neg)

    K_pos = 1 / T_pos
    K_neg = 1 / T_neg
         
    ##
    if Fp_form:
        F_pos = F_p
        F_neg = E_pos
    else:
        F_pos = F_p*E_pos
        F_neg = F_p*(1 - E_pos)

    return F_pos, F_neg, K_pos, K_neg 

#
#-------------------------------------------------------------------------------
def params_from_DIBEM(F_pos, F_neg, K_pos, K_neg):
    '''
    Starting with the derived parameters fitted in
    the 2CXM model, convert to the physiological parameters F_p, PS, ve and vep
    model given input physiological parameters
    [F_p, PS, v_e, v_p] = two_cx_params_model_to_phys(K_pos, K_neg, F_pos, F_neg)

    Inputs:
        F_pos, F_neg - scalars in 2CXM model IRF

        K_pos, K_neg - exponents in 2CXM model IRF

    Outputs:
        F_p - flow plasma rate

        PS - extraction flow

        v_e - extravascular, extracellular volume

        v_p - plasma volume
    '''
    _, F_pos, F_neg, K_pos, K_neg = helpers.check_param_shape(
        F_pos=F_pos, F_neg=F_neg, K_pos=K_pos, K_neg=K_neg
    )

    F_p = F_pos + F_neg
    E_pos = F_pos / F_p
    T_E = 1 / K_pos
    T_P = 1 / K_neg
    T_T = E_pos * (T_E-T_P) + T_P

    v_p = T_P*F_p
    v_e = T_T*F_p - v_p
    PS = v_e / T_E
    
    return F_p, PS, v_e, v_p

#
#---------------------------------------------------------------------------------
def concentration_from_model(aif:dce_aif.Aif, 
    Fp: np.array, PS: np.array, Ve: np.array, Vp: np.array, tau_a: np.array)->np.array:
    ''' 
    Compute concentration time-series of 2CXM from input
    paramaters. Note instead of re-implementing a bi-exponential
    model here, we call the DIBEM module to convert the 2CXM
    params to the bi-exponential parameters, and then call
    DIBEM's concentration_from_model

     Inputs:
        aif (Aif object, n_t): 
            object to store and resample arterial input function values
    
        Fp: np.array (1D n_samples) 
          flow plasma rate
    
        PS: np.array (1D n_samples) 
          extraction flow
    
        v_e: np.array (1D n_samples) 
          extravascular, extracellular volume
    
        v_p: np.array (1D n_samples) 
          plasma volume
    
        tau_a: np.array (1D n_samples) 
          tau_a times of arrival for conccentraion for Ca_t
    
     Returns:
       C_model (2D numpy array, n_samples x n_t) - Model concentrations at each time point for each 
       voxel computed from model paramaters
    
    '''

    #We derive the params in a standalone function now, this takes care of
    #checks on F_p, PS to choose the best form of derived parameters
    F_pos, F_neg, K_pos, K_neg  = params_to_DIBEM(
        Fp, PS, Ve, Vp)

    C_t = dibem.concentration_from_model(
        aif, F_pos, F_neg, K_pos, K_neg, 1.0, tau_a, 0)

    return C_t

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

    Notes:
      We can directly use the generic bi-expontential function with arterial mixing fraction f_a = 1.0

    '''
    return dibem.construct_LLS_matrix(Ctis_t, aif, 1.0, tau_a, 0)

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
        F_p, PS, v_e, v_p : float
            TK model parameters
    '''
    A_ = construct_LLS_matrix(Ctis_t, aif, tau_a)
    C_ = Ctis_t
    B_ = np.linalg.lstsq(A_, C_, rcond=None)[0]
    F_p = B_[3]
    T = B_[2] / (B_[0] * F_p)
    ba = np.sqrt(B_[1] * B_[1] - 4 * B_[0])
    T_e = (B_[1] + ba) / (2 * B_[0])
    T_p = (B_[1] - ba) / (2 * B_[0])

    v_p = T_p * F_p
    v_e = T * F_p - v_p
    PS = v_e / T_e

    return F_p, PS, v_e, v_p