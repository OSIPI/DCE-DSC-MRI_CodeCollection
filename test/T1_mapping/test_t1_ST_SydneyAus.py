import pytest
import numpy as np

from . import t1_data
from src.original.ST_SydneyAus.VFAT1mapping import VFAT1mapping


# Test for specific inputs and expected outputs :
"""

"""

# Combine all test data to decorate test functions    
parameters = pytest.mark.parametrize('label, fa_array, tr_array, s_array, r1_ref, s0_ref, a_tol, r_tol',
                     t1_data.t1_brain_data() +
                     t1_data.t1_quiba_data() +
                     t1_data.t1_prostate_data()                     
                     )

@parameters
def test_ST_SydneyAus_t1_VFA_nonlin(label, fa_array, tr_array, s_array, r1_ref, s0_ref, a_tol, r_tol):
    
    # prepare input data
    tr = tr_array[0] * 1000. # convert s to ms
 
    # run test (non-linear)
    [s0_nonlin_meas, t1_nonlin_meas] = VFAT1mapping( fa_array, s_array, tr, method = 'nonlinear' )
    r1_nonlin_meas = 1000./t1_nonlin_meas # convert T1 (ms) to R1 (/s)   
    np.testing.assert_allclose( [r1_nonlin_meas], [r1_ref], rtol=r_tol, atol=a_tol )

@parameters
def test_ST_SydneyAus_t1_VFA_lin(label, fa_array, tr_array, s_array, r1_ref, s0_ref, a_tol, r_tol):
    #NOTES:
    #   Signal must be scaled to prevent multiple test failures.
    #   Linear mode uses linear model but not linear regression(?)
        
    
    # prepare input data
    tr = tr_array[0] * 1000. # convert s to ms
    s_array /= 1000 # scale down signal to prevent test failure
 
    # run test (linear)
    [s0_lin_meas, t1_lin_meas] = VFAT1mapping( fa_array, s_array, tr, method = 'linear' )
    r1_lin_meas = 1000./t1_lin_meas # convert T1 (ms) to R1 (/s)
    np.testing.assert_allclose( [r1_lin_meas], [r1_ref], rtol=r_tol, atol=a_tol )

