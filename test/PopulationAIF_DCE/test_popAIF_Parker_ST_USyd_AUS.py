import os
import pytest
import numpy as np
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
from osipi_code_collection.original.ST_USyd_AUS.InputFunctions import AIF, variableAIF, arr_shift
from . import popAIF_data


# All tests will use the same arguments and same data...
arg_names = 'label, time, cb_ref_values, delay, a_tol, r_tol'
test_data = (
        popAIF_data.ParkerAIF_refdata() +
        popAIF_data.ParkerAIF_refdata_delay())

filename_prefix = ''

def setup_module(module):
    # initialize the logfiles
    global filename_prefix # we want to change the global variable
    os.makedirs('./test/results/PopulationAIF_DCE', exist_ok=True)
    filename_prefix = 'PopulationAIF_DCE/TestResults_PopAIF'
    log_init(filename_prefix, '_Parker_AIF_ST_USyd_AUS', ['label', 'time ('
                                                                   'us)', 'time_ref', 'aif_ref', 'cb_measured'])

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data.
# some test data fails as this code uses a shift of the original AIF in time and doesn't recalculate the values based on the new time array
@osipi_parametrize(arg_names, test_data, xf_labels=['delay_5.0s','delay_10.0s','delay_31.0s','delay_2.0s'])
def test_Parker_AIF_ST_Sydney_AUS(label, time, cb_ref_values, delay, a_tol,
                                r_tol):

    # prepare input data
    time = time*60 # time array is expected in seconds
    t0 = delay + time[1] # precontrast signal; t0 is expected to be in seconds; if no precontrast signal is expected, the value should be equal to the temp resolution
    tic = perf_counter()
    AIF_P = AIF(t0, time)
    exc_time = 1e6 * (perf_counter() - tic)  # measure execution time

    # log results
    row_data = []
    for t, ref, meas in zip(time, cb_ref_values, AIF_P):
        row_data.append([label, f"{exc_time:.0f}", t, ref, meas])
    log_results(filename_prefix, '_Parker_AIF_ST_USyd_AUS', row_data)

    np.testing.assert_allclose([AIF_P], [cb_ref_values], rtol=r_tol, atol=a_tol)
