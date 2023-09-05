import pytest
import os
import numpy as np
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
from . import t1_data
from osipi_code_collection.original.MJT_UoEdinburgh_UK.t1_fit import VFALinear, VFANonLinear, VFA2Points


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
    os.makedirs('./test/results/T1_mapping', exist_ok=True)
    filename_prefix = 'T1_mapping/TestResults_T1mapping'
    log_init(filename_prefix, '_MJT_Edinburgh_UK_t1_VFA_nonlin', ['label', 'time (us)', 'r1_ref', 'r1_measured'])
    log_init(filename_prefix, '_MJT_Edinburgh_UK_t1_VFA_lin', ['label', 'time (us)', 'r1_ref', 'r1_measured'])
    log_init(filename_prefix, '_MJT_Edinburgh_UK_t1_VFA_2fa', ['label', 'time (us)', 'r1_ref', 'r1_measured'])


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels = [])
def test_MJT_Edinburgh_UK_t1_VFA_nonlin(label, fa_array, tr_array, s_array, r1_ref, s0_ref, a_tol, r_tol):
    # NOTES:

    # prepare input data
    tr = tr_array[0]

    # run test (non-linear)
    tic = perf_counter()
    [s0_nonlin_meas, t1_nonlin_meas] = VFANonLinear(fa_array,tr).proc(s_array)
    exc_time = 1e6 * (perf_counter() - tic)
    r1_nonlin_meas = 1./t1_nonlin_meas
    log_results(filename_prefix, '_MJT_Edinburgh_UK_t1_VFA_nonlin', [[label, f"{exc_time:.0f}", r1_ref, r1_nonlin_meas]])  # log results
    np.testing.assert_allclose([r1_nonlin_meas], [r1_ref], rtol=r_tol, atol=a_tol)



# In the following test, we specify 1 case that is expected to fail...
@osipi_parametrize(arg_names, test_data, xf_labels = ['Pat5_voxel5_prostaat'])
def test_MJT_Edinburgh_UK_t1_VFA_lin(label, fa_array, tr_array, s_array, r1_ref, s0_ref, a_tol, r_tol):
    # NOTES:
    #   Expected fails: 1 low-SNR prostate voxel

    # prepare input data
    tr = tr_array[0]

    # run test (non-linear)
    tic = perf_counter()
    [s0_lin_meas, t1_lin_meas] = VFALinear(fa_array,tr).proc(s_array)
    exc_time = 1e6 * (perf_counter() - tic)
    r1_lin_meas = 1./t1_lin_meas
    log_results(filename_prefix, '_MJT_Edinburgh_UK_t1_VFA_lin', [[label, f"{exc_time:.0f}", r1_ref, r1_lin_meas]])  # log results
    np.testing.assert_allclose([r1_lin_meas], [r1_ref], rtol=r_tol, atol=a_tol)


# In the following test, we specify 1 case that is expected to fail...
@osipi_parametrize(arg_names, test_data, xf_labels = ['Pat5_voxel5_prostaat'])
def test_MJT_Edinburgh_UK_t1_VFA_2fa(label, fa_array, tr_array, s_array, r1_ref, s0_ref, a_tol, r_tol):
    # NOTES:
    #   Expected fails: 1 low-SNR prostate voxel

    # prepare input data
    tr = tr_array[0]

    # run test (non-linear)
    tic = perf_counter()
    [s0_2fa_meas, t1_2fa_meas] = VFA2Points(np.array([fa_array[0], fa_array[-1]]), tr).proc(np.array([s_array[0],
                                                                                                      s_array[-1]]))
    exc_time = 1e6 * (perf_counter() - tic)
    r1_2fa_meas = 1./t1_2fa_meas
    log_results(filename_prefix, '_MJT_Edinburgh_UK_t1_VFA_2fa', [[label, f"{exc_time:.0f}", r1_ref, r1_2fa_meas]])  #
    # log results
    np.testing.assert_allclose([r1_2fa_meas], [r1_ref], rtol=r_tol, atol=a_tol)