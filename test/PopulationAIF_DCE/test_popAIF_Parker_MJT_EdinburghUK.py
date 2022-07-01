import os
import numpy as np
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
import osipi_dce_dsc_repo.original.MJT_UoEdinburghUK.aifs as aifs
from . import popAIF_data


# All tests will use the same arguments and same data...
arg_names = 'label, time, cb_ref_values, delay, r_tol, a_tol'
test_data = (
        popAIF_data.ParkerAIF_refdata() +
        popAIF_data.ParkerAIF_refdata_delay())

filename_prefix = ''

def setup_module(module):
    # initialize the logfiles
    global filename_prefix # we want to change the global variable
    os.makedirs('./test/results/PopulationAIF_DCE', exist_ok=True)
    filename_prefix = 'PopulationAIF_DCE/TestResults_PopAIF'
    log_init(filename_prefix, '_Parker_AIF_MJT_EdinburghUK', ['label', 'time (us)', 'time_ref', 'aif_ref', 'cb_measured'])

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
# In the following test, we specify 5 cases that are expected to fail as this function expects the delay to be specified according to the temp resolution
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_Parker_AIF_MJT_EdinburghUK(label, time, cb_ref_values, delay, a_tol, r_tol):

    # prepare input data
    time = time*60  # time array is expected in seconds
    tstart = delay  # start time of AIF in seconds
    hct = 0  # hematocrit correction ignored for now

    tic = perf_counter()
    # Create the AIF object
    aif = aifs.Parker(hct, tstart)
    # for corresponding time array
    c_ap = aif.c_ap(time)
    exc_time = 1e6 * (perf_counter() - tic)  # measure execution time

    # log results
    row_data = []
    for t, ref, meas in zip(time, cb_ref_values, c_ap):
        row_data.append([label, f"{exc_time:.0f}", t, ref, meas])
    log_results(filename_prefix, '_Parker_AIF_MJT_EdinburghUK', row_data)

    np.testing.assert_allclose([c_ap], [cb_ref_values], rtol=r_tol, atol=a_tol)
