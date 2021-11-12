import os
import pytest
import numpy as np
from test.helpers import osipi_parametrize
import src.original.MJT_UoEdinburghUK.aifs as aifs
from . import popAIF_data


# All tests will use the same arguments and same data...
arg_names = 'label, time, cb_ref_values, delay, r_tol, a_tol'
test_data = (
        popAIF_data.ParkerAIF_refdata() +
        popAIF_data.ParkerAIF_refdata_delay())

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
# In the following test, we specify 5 cases that are expected to fail as this function expects the delay to be specified according to the temp resolution
@osipi_parametrize(arg_names, test_data, xf_labels=['delay_2.0s','delay_5.0s','delay_10.0s','delay_18.0s','delay_31.0s'])
def test_Parker_AIF_MJT_EdinburghUK(label, time, cb_ref_values, delay, a_tol, r_tol):

    # prepare input data
    time = time*60  # time array is expected in seconds
    tstart = delay  # start time of AIF in seconds
    hct = 0  # hematocrit correction ignored for now

    # Create the AIF object
    aif = aifs.parker(hct, tstart)
    # for corresponding time array
    c_ap = aif.c_ap(time)
    np.testing.assert_allclose([c_ap], [cb_ref_values], rtol=r_tol, atol=a_tol)
