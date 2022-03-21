import os
import pytest
import numpy as np
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
from osipi_code_collection.original.PvH_NKI_NL.AIF.PopulationAIF import ParkerAIF
from . import popAIF_data


# All tests will use the same arguments and same data...
arg_names = 'label, time, cb_ref_values, delay, a_tol, r_tol'
test_data = popAIF_data.ParkerAIF_refdata()

filename_prefix = ''

def setup_module(module):
    # initialize the logfiles
    global filename_prefix # we want to change the global variable
    os.makedirs('./test/results/PopulationAIF_DCE', exist_ok=True)
    filename_prefix = 'PopulationAIF_DCE/TestResults_PopAIF'
    log_init(filename_prefix, '_Parker_AIF_PvH_NKI_NL', ['label', 'time (us)', 'time_ref', 'aif_ref', 'cb_measured'])

# this function does not have an option to specify the delay of the aif, so the ParkerAIF_refdata_delay() are ignored here
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_Parker_AIF_PvH_NKI_NL(label, time, cb_ref_values, delay, a_tol, r_tol):

    # prepare input data
    # convert time to seconds
    time = time*60

    tic = perf_counter()
    AIF_P = ParkerAIF(time)
    exc_time = 1e6 * (perf_counter() - tic)  # measure execution time

    # log results
    row_data = []
    for t, ref, meas in zip(time, cb_ref_values, AIF_P):
        row_data.append([label, f"{exc_time:.0f}", t, ref, meas])
    log_results(filename_prefix, '_Parker_AIF_PvH_NKI_NL', row_data)

    # actual testing
    np.testing.assert_allclose([AIF_P], [cb_ref_values], rtol=r_tol, atol=a_tol)
