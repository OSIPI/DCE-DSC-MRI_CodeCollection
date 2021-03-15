import pytest
import numpy as np
from math import pi
from lmfit import Model
import pandas as pd

import t1_data
from src.original.ST_SydneyAus.VFAT1mapping import spgr_linear, spgr_nonlinear, VFAT1mapping


# Test for specific inputs and expected outputs :
"""

"""

@t1_data.parameters
def test_ST_SydneyAus_t1_VFA_nonlin(label, fa_array, tr_array, s_array, r1_ref, s0_ref):
    #TODO: document
    
    # prepare input data
    tr = tr_array[0] * 1000. # convert s to ms
    t1_ref = (1./r1_ref) * 1000. # convert R1 in /s to T1 in ms
 
    # run test (non-linear)
    [s0_nonlin_meas, t1_nonlin_meas] = VFAT1mapping( fa_array, s_array, tr, method = 'nonlinear' )
    r1_nonlin_meas = 1000./t1_nonlin_meas    
    np.testing.assert_allclose( [s0_nonlin_meas, r1_nonlin_meas], [s0_ref, r1_ref], rtol=0.15, atol=0 )

@t1_data.parameters
def test_ST_SydneyAus_t1_VFA_lin(label, fa_array, tr_array, s_array, r1_ref, s0_ref):
    #TODO: document
    
    # prepare input data
    tr = tr_array[0] * 1000. # convert s to ms
    t1_ref = (1./r1_ref) * 1000. # convert R1 in /s to T1 in ms
 
    # run test (linear)
    [s0_lin_meas, t1_lin_meas] = VFAT1mapping( fa_array, s_array, tr, method = 'linear' )
    r1_lin_meas = 1000./t1_lin_meas    
    np.testing.assert_allclose( [s0_lin_meas, r1_lin_meas], [s0_ref, r1_ref], rtol=0.15, atol=0 )

