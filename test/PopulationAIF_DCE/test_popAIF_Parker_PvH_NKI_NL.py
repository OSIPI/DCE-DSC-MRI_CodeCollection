import os
import pytest
import numpy as np
from test.helpers import osipi_parametrize
from src.original.PvH_NKI_NL.AIF.PopulationAIF import ParkerAIF
from . import popAIF_data


# All tests will use the same arguments and same data...
arg_names = 'label, time, cb_ref_values, delay, a_tol, r_tol'
test_data = popAIF_data.ParkerAIF_refdata()

# this function does not have an option to specify the delay of the aif, so the ParkerAIF_refdata_delay() are ignored here
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_Parker_AIF_PvH_NKI_NL(label, time, cb_ref_values, delay, a_tol, r_tol):

    # prepare input data
    # convert time to seconds
    time = time*60
    AIF_P = ParkerAIF(time)

    np.testing.assert_allclose([AIF_P], [cb_ref_values], rtol=r_tol, atol=a_tol)
