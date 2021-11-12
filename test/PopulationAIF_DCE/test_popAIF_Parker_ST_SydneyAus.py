import os
import pytest
import numpy as np
from test.helpers import osipi_parametrize
from src.original.ST_USydAUS.InputFunctions import AIF, variableAIF, arr_shift
from . import popAIF_data


# All tests will use the same arguments and same data...
arg_names = 'label, time, cb_ref_values, delay, r_tol, a_tol'
test_data = (
        popAIF_data.ParkerAIF_refdata() +
        popAIF_data.ParkerAIF_refdata_delay())

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_Parker_AIF_ST_SydneyAus(label, time, cb_ref_values, delay, a_tol, r_tol):

    # prepare input data
    time = time*60 # time array is expected in seconds
    t0 = delay + time[1] # precontrast signal; t0 is expected to be in seconds; if no precontrast signal is expected, the value should be equal to the temp resolution
    AIF_P = AIF(t0, time)
    np.testing.assert_allclose([AIF_P], [cb_ref_values], rtol=r_tol, atol=a_tol)
