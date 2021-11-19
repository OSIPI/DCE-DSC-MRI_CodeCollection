import os
import pytest
import numpy as np
from test.helpers import osipi_parametrize
import src.original.MB_QBI_UoManchesterUK.QbiPy.dce_models.dce_aif as dce_aif
from . import popAIF_data


# All tests will use the same arguments and same data...
arg_names = 'label, time, cb_ref_values, delay, r_tol, a_tol'
test_data = (
        popAIF_data.ParkerAIF_refdata() +
        popAIF_data.ParkerAIF_refdata_delay())

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
# In the following test, we specify 5 cases that are expected to fail as this function expects the delay to be specified according to the temp resolution
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_Parker_AIF_MB_QBI_UoManchesterUK(label, time, cb_ref_values, delay, a_tol, r_tol):

    # prepare input data
    # time array is expected in minutes, so no changes needed.
    hct = 0  # for now ignore hematocrit correction to obtain Cb values
    #time = time*60  # put in seconds
    # Create the AIF object
    aif = dce_aif.Aif(times=time, hct=hct)
    aif_values = aif.base_aif_[0,]

    np.testing.assert_allclose([aif_values], [cb_ref_values], rtol=r_tol, atol=a_tol)
