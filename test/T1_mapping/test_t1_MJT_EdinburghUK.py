import pytest
import numpy as np

from . import t1_data
from src.original.MJT_EdinburghUK.t1 import fit_vfa_nonlinear, fit_vfa_linear

# Test for specific inputs and expected outputs :
"""

"""

# combine all test data to decorate test functions    
parameters = pytest.mark.parametrize('label, fa_array, tr_array, s_array, r1_ref, s0_ref',
                     t1_data.t1_brain_data() +
                     t1_data.t1_quiba_data() +
                     t1_data.t1_prostate_data()                     
                     )

@parameters
def test_MJT_EdinburghUK_t1_VFA_nonlin(label, fa_array, tr_array, s_array, r1_ref, s0_ref):
    
    # prepare input data
    tr = tr_array[0]
    fa_array_rad = fa_array * np.pi/180.
    
    # run test (non-linear)
    [s0_nonlin_meas, t1_nonlin_meas] = fit_vfa_nonlinear(s_array,fa_array_rad,tr)

    r1_nonlin_meas = 1./t1_nonlin_meas    
    np.testing.assert_allclose( [s0_nonlin_meas, r1_nonlin_meas], [s0_ref, r1_ref], rtol=0.15, atol=0 )


@parameters
def test_MJT_EdinburghUK_t1_VFA_lin(label, fa_array, tr_array, s_array, r1_ref, s0_ref):
    
    # prepare input data
    tr = tr_array[0]
    fa_array_rad = fa_array * np.pi/180.
    
    # run test (non-linear)
    [s0_lin_meas, t1_lin_meas] = fit_vfa_linear(s_array,fa_array_rad,tr)

    r1_lin_meas = 1./t1_lin_meas    
    np.testing.assert_allclose( [s0_lin_meas, r1_lin_meas], [s0_ref, r1_ref], rtol=0.15, atol=0 )