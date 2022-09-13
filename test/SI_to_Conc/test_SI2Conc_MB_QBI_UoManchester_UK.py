import pytest
import os
import numpy as np
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
from . import SI2Conc_data
from osipi_code_collection.original.MB_QBI_UoManchester_UK.QbiPy.dce_models.tissue_concentration import signal_to_concentration




# All tests will use the same arguments and same data...
arg_names = 'label', 'fa', 'tr', 'T1base', 'BLpts', 'r1', 's_array', 'conc_array', 'a_tol', 'r_tol'
test_data = SI2Conc_data.SI2Conc_data()

filename_prefix = ''

def setup_module(module):
    # initialize the logfiles
    global filename_prefix # we want to change the global variable
    os.makedirs('./test/results/SI_to_Conc', exist_ok=True)
    filename_prefix = 'SI_to_Conc/TestResults_SI2Conc'
    log_init(filename_prefix, '_MB_UoManchester_UK', ['label', 'time (us)', 'conc_ref', 'conc_meas'])

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels = [])
def test_MB_UoManchester_sig_to_conc(label, fa, tr, T1base, BLpts, r1, s_array, conc_array, a_tol, r_tol):
    # Note: the first signal value is not used for baseline estimation,
    # and the first C value is not logged or assessed

    ##Prepare input data
    
    #The code needs a baseline signal to calculate M0 from, so this will be the mean signal from 1 to BL points as used in the orginal data
    S0=np.mean(s_array[1:BLpts])

    #The code used to make the test data doesn't include the first point on the curve, so remove it here from s and conc and reduce BLpts by 1
    s_array=s_array[1:]
    conc_array=conc_array[1:]
    BLpts=BLpts-1

    #Relaxivity units should be per ms rather than per s for this code, so multiply r1 by 1000
    r1=r1*1000

    # run test
    tic = perf_counter()
    conc_curve = signal_to_concentration(s_array, T1base, S0, fa, tr, r1, BLpts)[0]
    exc_time = 1e6 * (perf_counter() - tic)

    # log results
    row_data = []
    for ref, meas in zip(conc_array, conc_curve):
        row_data.append([label, f"{exc_time:.0f}", ref, meas])
    log_results(filename_prefix, '_MB_UoManchester_UK', row_data)

    np.testing.assert_allclose( conc_curve, conc_array, rtol=r_tol, atol=a_tol )

