import pytest
import numpy as np

from ..helpers import osipi_parametrize
from . import SI2Conc_data
from osipi_code_collection.original.LCB_BNI.dce import signal_to_conc


# All tests will use the same arguments and same data...
arg_names = 'label', 'fa', 'tr', 'T1base', 'BLpts', 'r1', 's_array', 'conc_array', 'a_tol', 'r_tol'
test_data = SI2Conc_data.SI2Conc_data()


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels = [])
def test_LCB_BNI_sig_to_conc(label, fa, tr, T1base, BLpts, r1, s_array, conc_array, a_tol, r_tol):

    ##Prepare input data

    conc_curve = signal_to_conc(s_array, [1,BLpts-1], T1base, tr, fa, r1)


    np.testing.assert_allclose( conc_curve, conc_array, rtol=r_tol, atol=a_tol )

