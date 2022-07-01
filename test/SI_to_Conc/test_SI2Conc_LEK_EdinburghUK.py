import pytest
import os
import numpy as np
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
from . import SI2Conc_data
from osipi_dce_dsc_repo.original.LEK_UoEdinburghUK.SignalToConcentration import SI2Conc


# All tests will use the same arguments and same data...
arg_names = 'label', 'fa', 'tr', 'T1base', 'BLpts', 'r1', 's_array', 'conc_array', 'a_tol', 'r_tol'
test_data = SI2Conc_data.SI2Conc_data()

filename_prefix = ''

def setup_module(module):
    # initialize the logfiles
    global filename_prefix # we want to change the global variable
    os.makedirs('./test/results/SI_to_Conc', exist_ok=True)
    filename_prefix = 'SI_to_Conc/TestResults_SI2Conc'
    log_init(filename_prefix, '_LEK_UoEdinburgh', ['label', 'time (us)', 'conc_ref', 'conc_meas'])

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels = [])
def test_LEK_UoEdinburghUK_SI2Conc(label, fa, tr, T1base, BLpts, r1, s_array, conc_array, a_tol, r_tol):
    # Note: the first signal value is not used for baseline estimation,
    # and the first C value is not logged or assessed

    #Prepare input data
    #Nothing to do for this function
    
    # run test
    tic = perf_counter()
    conc_curve = SI2Conc.SI2Conc(s_array,tr,fa,T1base,BLpts,S0=None)
    exc_time = 1e6 * (perf_counter() - tic)

    # log results
    row_data = []
    for ref, meas in zip(conc_array[1:], conc_curve[1:]/r1):
        row_data.append([label, f"{exc_time:.0f}", ref, meas])
    log_results(filename_prefix, '_LEK_UoEdinburgh', row_data)

    # testing
    conc_array=conc_array*r1 # This function doesn't include r1, so multiply it out before testing
    np.testing.assert_allclose( [conc_curve[1:]], [conc_array[1:]], rtol=r_tol, atol=a_tol)

