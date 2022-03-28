import pytest
import os
import numpy as np
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
from . import SI2Conc_data
from osipi_code_collection.original.ST_USydAUS.signals2conc import signals2conc


# All tests will use the same arguments and same data...
arg_names = 'label', 'fa', 'tr', 'T1base', 'BLpts', 'r1', 's_array', 'conc_array', 'a_tol', 'r_tol'
test_data = SI2Conc_data.SI2Conc_data()

filename_prefix = ''

def setup_module(module):
    # initialize the logfiles
    global filename_prefix # we want to change the global variable
    os.makedirs('./test/results/SI_to_Conc', exist_ok=True)
    filename_prefix = 'SI_to_Conc/TestResults_SI2Conc'
    log_init(filename_prefix, '_ST_USydAus', ['label', 'time (us)', 'conc_ref', 'conc_meas'])


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels = [])
def test_ST_USydAUS_signals2conc(label, fa, tr, T1base, BLpts, r1, s_array, conc_array, a_tol, r_tol):
    ##Prepare input data
    #The code used to make the test data doesn't include the first point on the curve, so remove it here from s and conc and reduce BLpts by 1
    s_array=s_array[1:]
    conc_array=conc_array[1:]
    BLpts=BLpts-1
    # This function takes time as an argument but only uses its length, so construct a dummy array
    time=np.zeros_like(s_array)
    
    # run test
    tic = perf_counter()
    conc_curve = signals2conc(time, s_array, fa, tr, 1/T1base, r1, BLpts)
    exc_time = 1e6 * (perf_counter() - tic)

    # log results
    row_data = []
    for ref, meas in zip(conc_array, conc_curve):
        row_data.append([label, f"{exc_time:.0f}", ref, meas])
    log_results(filename_prefix, '_ST_USydAus', row_data)

    np.testing.assert_allclose( [conc_curve], [conc_array], rtol=r_tol, atol=a_tol )

