import os
import pytest
import numpy as np
from test.helpers import osipi_parametrize
from src.original.PvH_NKI_NL.AIF.PopulationAIF import GeorgiouAIF
from . import popAIF_data


# All tests will use the same arguments and same data...
arg_names = 'label, time, cb_ref_values, r_tol, a_tol'
test_data = popAIF_data.GeorgiouAIF_refdata()


@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_Georgiou_AIF_PvH_NKI_NL(label, time, cb_ref_values, a_tol, r_tol):

    # prepare input data
    # convert time to seconds
    time = time*60
    AIF_G = GeorgiouAIF(time)

    np.testing.assert_allclose([AIF_G], [cb_ref_values], rtol=r_tol, atol=a_tol)
