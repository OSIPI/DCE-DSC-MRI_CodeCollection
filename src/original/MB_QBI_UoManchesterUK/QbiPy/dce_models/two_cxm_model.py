'''
Module for working with the two-compartment exchange model (2CXM). 
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

---------------------- 2CXM conversions ----------------------------------
 2CXM model is bi-exponential, with  concentration computed as
   C(t) = [ F_pos.exp(-t.K_pos) + F_neg.exp(-t.K_neg) ] ** Ca(t)

 Where
   K_pos = K_sum + K_root
   K_neg = K_sum - K_root
 
   E_pos = (T_neg - Kb) / (T_neg + T_pos)
   F_pos = F_p.E_pos
   F_neg = F_p.(1 - E_pos)

 Derived from

   Kp = (F_p + PS) / v_p
   Ke = PS / v_ecs
   Kb = F_p / v_p
   K_sum = 0.5*(Kp + Ke)
   K_root = 0.5* sqrt( (Kp + Ke).^2 - 4*Ke *Kb)

 Where

   F_p - flow plasma rate
   PS = extraction flow
   v_e - extra cellular extra vascular volume
   v_p - plasma vlume
 
 See paper: Phys Med Bio. 201055:6431-6643
   "Error estimation for perfusion parameters obtained using the 
   two-compartment exchange model in dynamic contrast-enhanced MRI: a simulation study"
   R Luypaert, S Sourbron, S Makkat and J de Mey.

'''

import numpy as np
from ...QbiPy.dce_models import dce_aif, dibem
from ...QbiPy import helpers

#
#-------------------------------------------------------------------------------
def params_to_DIBEM(F_p, PS, v_e, v_p, Fp_form=False):
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
          extravascular, extracellular volume fraction

        v_p: np.array (1D n_samples) 
          plasma volume fraction

        Fp_form: bool 
          if true convert to alternative form Fp.E+.exp(-tK+) + Fp.(1-E+).exp(-t.K-)

    Outputs:
        F_pos, F_neg - multipliers in model IRF

        K_pos, K_neg - exponents in model IRF

    Notes:

    We can derive the params in a couple of ways, which remain stable under
    different conditions of ve, vp, PS and FP

    The first way is as derived in the Sourbron 2011 MRM paper, which is valid
    except when PS = 0 or FP = 0. The second method is as derived in Luypaert 
    paper 2010 paper. It works when PS or FP = 0, but doesn't like ve or vp = 0
    '''
    dims_sz, F_p, PS, v_e, v_p = helpers.check_param_shape(
        F_p=F_p, PS=PS, v_e=v_e, v_p=v_p
    )

    method1 = (PS > 0) & (F_p > 0) & ((v_e + v_p) > 0)
    method2 = ~method1

    #We're assuming all params have been passed in the same size, not doing any
    #error checks here
    if np.any(method1):
        K_pos = np.zeros(dims_sz)
        K_neg = np.zeros(dims_sz)
        E_pos = np.zeros(dims_sz)

        ## Method 1: Sourbron 2011
        #First derive the secondary parameters from the input Pk parameters
        E = PS[method1] / (PS[method1] + F_p[method1]) #Extraction fraction
        e = v_e[method1] / (v_p[method1] + v_e[method1]) #Extractcellular fraction

        tau = (E - E*e + e) / (2*E)
        tau_root = np.sqrt(1 - 4*(E*e*(1-E)*(1-e)) / ((E - E*e + e)**2) )
        tau_pos = tau * (1 + tau_root)
        tau_neg = tau * (1 - tau_root)

        K_pos[method1] = F_p[method1] / ((v_p[method1] + v_e[method1])*tau_neg)
        K_neg[method1] = F_p[method1] / ((v_p[method1] + v_e[method1])*tau_pos)

        E_pos[method1] = (tau_pos - 1) / (tau_pos - tau_neg)

    ## Method 2
    if np.any(method2):
        Kp = (F_p[method2] + PS[method2]) / v_p[method2]
        Ke = PS[method2] / v_e[method2]
        Kb = F_p[method2] / v_p[method2]

        K_sum = 0.5*(Kp + Ke)
        K_root = 0.5* np.sqrt( (Kp + Ke)**2 - 4*Ke *Kb)
        K_pos[method2] = K_sum - K_root
        K_neg[method2] = K_sum + K_root

        E_pos[method2] = (K_neg[method2] - Kb) / (K_neg[method2] - K_pos[method2])
         
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

    #We derive the params based on 2009 Sourbron paper
    F_p = F_pos
    E_neg = (1 - F_neg)

    T_B = 1 / (K_pos - E_neg * (K_pos - K_neg))
    T_E = 1 / (T_B * K_pos * K_neg)
    T_P_inv = K_pos + K_neg - 1 / T_E

    v_p = F_p * T_B
    PS = F_p * (T_B * T_P_inv - 1)
    v_e = PS * T_E 

    apply_tm = (K_pos==0) & (F_pos==0)
    if np.any(apply_tm):
        PS[apply_tm] = np.NaN
        v_p[apply_tm] = 0
        v_e[apply_tm] = F_p[apply_tm] / K_neg[apply_tm]
    
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

     Parameters:
       aif (Aif object, n_t): object to store and resample arterial input function values (1 for each time point)
    
     Parameters:
        Fp: np.array (1D n_samples) 
          flow plasma rate
    
        PS: np.array (1D n_samples) 
          extraction flow
    
        v_e: np.array (1D n_samples) 
          extravascular, extracellular volume
    
        v_p: np.array (1D n_samples) 
          plasma volume
    
        tau_a: np.array (1D n_samples) 
          delay times for conccentraion for Ca_t
    
     Returns:
       C_model (2D numpy array, n_samples x n_t) - Model concentrations at each time point for each 
       voxel computed from model paramaters
    
    '''

    #We derive the params in a standalone function now, this takes care of
    #checks on FP, PS to choose the best form of derived parameters
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
    T = B_[2] / (B_[0]*F_p)
    T_e = B_[1] / B_[0] - T
    T_p = 1 / (B_[0]*T_e)

    v_p = T_p * F_p
    v_e = T * F_p - v_p
    PS = v_e / T_e

    return F_p, PS, v_e, v_p
    
