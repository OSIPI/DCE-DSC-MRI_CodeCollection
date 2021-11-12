import pytest
import numpy as np

from ..helpers import osipi_parametrize, log_init, log_results
from . import t1_data
from osipi.original.OG_MO_AUMC_ICR_RMH.ExtendedTofts.DCE import R1_two_fas


# All tests will use the same arguments and same data...
arg_names = 'label, fa_array, tr_array, s_array, r1_ref, s0_ref, a_tol, r_tol'
test_data = (
    t1_data.t1_brain_data() +
    t1_data.t1_quiba_data() +
    t1_data.t1_prostate_data()
    )

filename_prefix = ''

def setup_module(module):
    # initialize the logfiles
    global filename_prefix # we want to change the global variable
    filename_prefix = 'TestResults_T1mapping'
    log_init(filename_prefix, '_test_OG_MO_AUMC_ICR_RHM_t1_VFA_2fa', ['label', 'r1_ref', 'r1_measured'])


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
# In the following test, we specify 1 case that is expected to fail...
@osipi_parametrize(arg_names, test_data, xf_labels = ['Pat5_voxel5_prostaat'])
def testOG_MO_AUMC_ICR_RMH_t1_VFA_2fa(label, fa_array, tr_array, s_array, r1_ref, s0_ref, a_tol, r_tol):
    # NOTES:
    #   Code requires signal array with min 2 dimensions (including FA)
    #   Expected fails: 1 low-SNR prostate voxel
    
    # prepare input data
    tr = tr_array[0]
    # use first and last FA only
    s_array_trimmed = np.array([s_array[[0, -1]]]) # use first and last FA, make 2D
    fa_array_rad = fa_array[[0, -1]] * np.pi/180. # use first and last FA only
    
    # run test (2 flip angle)
    r1_2fa_meas = R1_two_fas(s_array_trimmed, fa_array_rad,tr)[0]
    np.testing.assert_allclose([r1_2fa_meas], [r1_ref], rtol=r_tol, atol=a_tol)
    log_results(filename_prefix, '_test_OG_MO_AUMC_ICR_RHM_t1_VFA_2fa', [label, r1_ref, r1_2fa_meas])