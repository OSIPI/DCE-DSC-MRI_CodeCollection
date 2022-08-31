'''
Module for working with the active uptake and efflux model (AUEM). This has
a bi-exponential IRF, and uses the dibem model to compute a forward model.

All times are assumed to be in minutes.

The AIF must be a QbiPy AIF object (see dce_aif). However if you have a set of AIF values (Ca_t)
and associated dynamic times (t), it is trivial to create an AIF object:

aif = dce_aif.Aif(times = t, base_aif=Ca_t, aif_type=ARRAY)

The remaining model parameters can either be input as scalars, or 1D numpy arrays. The two forms
can be mixed, but any paramaters set as arrays must be the same length.

Code for converting AUEM parameters to DIBEM form is defined below.

---------------------- AUEM conversions ----------------------------------
Concentration model equation
   Cl_t = F_p.(E_i.exp(-t/Ti) / (1 - T_e/T_i) + (1 - E_i/(1 - T_e / T_i)).exp(-t/Te)) * Cp_t

 Where
   Cp_t = (f_a.Ca_t + f_v.Cv_t) / (1 - Hct)

   F_p - flow plasma rate
   T_e = v_ecs / (F_p + k_i) - extracellular mean transit time
   T_i = vi / kef - intracellular mean transit time
   E_i = ki / (Fp + ki) - the hepatic uptake fraction
   f_a - the arterial fraction
   f_v = 1 - fa - estimate of hepatic portal venous fraction
   v_i = 1 - v_ecs - estimate of intracellular volume
 
 See paper: Invest Radiol. 2017 Feb52(2):111-119. doi: 10.1097/RLI.0000000000000316.
   "Quantitative Assessment of Liver Function Using Gadoxetate-Enhanced Magnetic Resonance Imaging: 
   Monitoring Transporter-Mediated Processes in Healthy Volunteers"
   Georgiou L1, Penny J, Nicholls G, Woodhouse N, Bl FX, Hubbard Cristinacce PL, Naish JH.

'''

import warnings
import numpy as np
from QbiPy.dce_models import dce_aif, dibem
from QbiPy import helpers

#
#-------------------------------------------------------------------------------
def params_to_DIBEM(F_p, v_ecs, k_i, k_ef, using_Fp=False):
    '''compute the derived parameters for the AUEM given input physiological parameters
   [K_pos, K_neg, F_pos, F_neg] = active_params_phys_to_model(F_p, v_e, k_i, k_ef)

    Inputs:
        F_p - flow plasma rate

        v_ecs - extra-cellular space (v_i = 1 - v_ecs)

        k_i - active-uptake rate

        k_ef - efflux rate

    Outputs:
        F_pos, F_neg - scalars in model IRF
    
        K_pos, K_neg - exponents in model IRF
    '''
    _, F_p, v_ecs, k_i, k_ef = helpers.check_param_shape(
        F_p=F_p, v_ecs=v_ecs, k_i=k_i, k_ef=k_ef
    )  

    #Compute derived parameters from input parameters
    T_e = v_ecs / (F_p + k_i) # extracellular mean transit time
    v_i = 1 - v_ecs # - etsimate of intracellular volume
    T_i = v_i / k_ef # intracellular mean transit time
    E_i = k_i / (F_p + k_i) # the hepatic uptake fraction

    #This can also be precomputed
    E_pos = E_i / (1 - T_e/T_i)

    K_neg = 1 / T_e
    K_pos = 1 / T_i

    if using_Fp:
        F_pos = F_p
        F_neg = E_pos
    else:
        F_pos = F_p*E_pos
        F_neg = F_p*(1 - E_pos)

    return F_pos, F_neg, K_pos, K_neg, 

#
#-------------------------------------------------------------------------------
def params_from_DIBEM(F_pos, F_neg, K_pos, K_neg, 
    using_Fp=False, warn_mode = 'warn'):
    '''
    Starting with the derived parameters fitted in
    the IRF-3 model, convert to the physiological parameters F_p, v_ecs, k_i
    and k_ef
    model given input physiological parameters
    [F_p, v_ecs, k_i, k_ef] = active_params_model_to_phys(K_pos, K_neg, F_pos, F_neg)

    Inputs:
        F_pos, F_neg - scalars in 2CXM model IRF

        K_pos, K_neg - exponents in 2CXM model IRF

    Outputs:
        F_p - flow plasma rate

        v_ecs - extra-cellular space (v_i = 1 - v_ecs)

        k_i - active-uptake rate

        k_ef - efflux rate

    Concentration model equation
    Cl_t = F_p.(E_i.exp(-t/Ti) / (1 - T_e/T_i) + (1 - E_i/(1 - T_e / T_i)).exp(-t/Te)) * Cp_t

    Where
    Cp_t = (f_a.Ca_t + f_v.Cv_t) / (1 - Hct)

    F_p - flow plasma rate
    T_e = v_ecs / (F_p + k_i) - extracellular mean transit time
    T_i = vi / kef - intracellular mean transit time
    E_i = ki / (Fp + ki) - the hepatic uptake fraction
    f_a - the arterial fraction
    f_v = 1 - fa - estimate of hepatic portal venous fraction
    v_i = 1 - v_ecs - estimate of intracellular volume
    
    See paper: Invest Radiol. 2017 Feb52(2):111-119. doi: 10.1097/RLI.0000000000000316.
    "Quantitative Assessment of Liver Function Using Gadoxetate-Enhanced Magnetic Resonance Imaging:"
    Georgiou L1, Penny J, Nicholls G, Woodhouse N, Bl FX, Hubbard Cristinacce PL, Naish JH.'''
    _, F_pos, F_neg, K_pos, K_neg = helpers.check_param_shape(
        F_pos=F_pos, F_neg=F_neg, K_pos=K_pos, K_neg=K_neg
    )

    #First get F_p from F_pos and F_neg
    if not using_Fp:
        F_p = F_pos + F_neg
        E_pos = F_pos / F_p
    else:
        F_p = F_pos
        E_pos = F_neg

    #Derivation is only valid for K_pos < K_neg. If not, the swapping
    #F_pos, K_pos for F_neg, K_neg will generate valid active parameters (and
    #an indentical concentration time series due to symmetry of the
    #bi-exponential). User defines whether swap with warning, quietly or force
    #an error if invalid voxels found
    swap_idx = K_pos > K_neg
    if np.any(swap_idx):
        if warn_mode == 'warn':
            warnings.warn(
                f'K_pos > K_neg for {np.sum(swap_idx)} of {swap_idx.size} voxels. Switching these voxels')
        elif warn_mode == 'error':
            raise RuntimeError(
                f'K_pos > K_neg for {np.sum(swap_idx)} of {swap_idx.size} voxels. ' 
                'Run with warn_mode = ''quiet'' or ''warn to switch these voxels.')
        elif warn_mode == 'quiet':
            #do nothing
            pass
        else:
            raise ValueError('Warn mode {warn_mode} not recognised. Must be ''warn'', ''quiet'' or ''error''')
        
        if not using_Fp:
            #F_p doesn't change it is the sum of F_pos and F_neg
            #E_pos needs to remade from F_neg for the swapped indices
            E_pos[swap_idx] = F_neg[swap_idx] / F_p[swap_idx]
        else:
            #F_p doesn't change, E_pos needs negating
            E_pos[swap_idx] = 1 - E_pos[swap_idx]
        
        #K_pos and K_neg are just a straight swap
        K_pos_swap = K_pos[swap_idx]
        K_pos[swap_idx] = K_neg[swap_idx]   
        K_neg[swap_idx] = K_pos_swap

    #Now derive Te, Ti and Ei
    Te = 1 / K_neg
    Ti = 1 / K_pos
    Ei = E_pos * (1 - Te / Ti)

    #Can solve for k_i in terms of F_p and Ei
    k_i = Ei * F_p / (1 - Ei)

    #Solve for v_ecs in terms of Te, F_p and K-i
    v_ecs = Te * (F_p + k_i)

    #Finally solve for k_ef in terms of v_ecs and Ti
    k_ef = (1 - v_ecs) / Ti
    return F_p, v_ecs, k_i, k_ef

#
#---------------------------------------------------------------------------------
def concentration_from_model(aif:dce_aif.Aif, 
    Fp: np.array, PS: np.array, Ve: np.array, Vp: np.array, 
    f_a:np.array, tau_a: np.array, tau_v:np.array)->np.array:
    ''' 
    Compute concentration time-series of 2CXM from input
    paramaters. Note instead of re-implementing a bi-exponential
    model here, we call the DIBEM module to convert the 2CXM
    params to the bi-exponential parameters, and then call
    DIBEM's concentration_from_model

     Parameters:
       aif (Aif object, n_t): object to store and resample arterial input function values (1 for each time point)
    
     Parameters:
        Fp: np.array (1D n_samples)
            flow plasma rate

        v_ecs: np.array (1D n_samples)
            extra-cellular volume fraction

        k_i: np.array (1D n_samples)
            uptake rate constant

        k_ef: np.array (1D n_samples)
            efflux rate constant

        f_a: np.array (1D n_samples)
            Arterial mixing fraction, final plasma input is Cp(t) = f_a*Ca(t) + (1-f_a)*Cv(t)

        tau_a: np.array (1D n_samples)
            offset times of arrival for conccentraion for Ca_t

        tau_v: np.array (1D n_samples)
            offset times of arrival for conccentraion for Cv_t
    
     Returns:
       C_model (2D numpy array, n_t x n_vox) - Model concentrations at each time point for each 
       voxel computed from model paramaters
    
    '''

    #We derive the params in a standalone function now, this takes care of
    #checks on FP, PS to choose the best form of derived parameters
    F_pos, F_neg, K_pos, K_neg  = params_to_DIBEM(
        Fp, PS, Ve, Vp)

    C_t = dibem.concentration_from_model(
        aif, F_pos, F_neg, K_pos, K_neg, f_a, tau_a, tau_v)

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

    Notes:
      We can directly use the generic bi-expontential function

    '''
    return dibem.construct_LLS_matrix(Ctis_t, aif, f_a, tau_a, tau_v)

#
#---------------------------------------------------------------------------
def solve_LLS(Ctis_t:np.array, aif:dce_aif.Aif, f_a:float, tau_a:float, tau_v:float):
    '''
    Solve model parameters for a single tissue time-series using LLS
	
	Inputs:
        Ct_sig: np.array (num_times)
            time-series of signal derived CA concentration

        aif (Aif object, num_times): 
            object to store and resample arterial input function values (1 for each time point)
    
        f_a: float
            Arterial mixing fraction, final plasma input is Cp(t) = f_a*Ca(t) + (1-f_a)*Cv(t)
            
        tau_a: float
            offset times of arrival for conccentraion for Ca_t

        tau_v: float
            offset times of arrival for conccentraion for Cv_t

    Outputs:
        F_p, v_ecs, k_i, k_ef : float
            TK model parameters

    Notes:
        Need to complete this!
    '''
    A_ = construct_LLS_matrix(Ctis_t, aif, f_a, tau_a, tau_v)
    C_ = Ctis_t
    B_ = np.linalg.lstsq(A_, C_, rcond=None)[0]
    F_p = B_[3]
    T = B_[2] / (B_[0]*F_p)
    T_e = B_[1] / B_[0] - T
    T_p = 1 / (B_[0]*T_e)
    v_ecs = T_p * F_p

    #TODO 
    k_i = 0 
    k_ef = 0

    return F_p, v_ecs, k_i, k_ef