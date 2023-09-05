import pytest
import numpy as np

from ..helpers import osipi_parametrize
from . import SI2Conc_data
from osipi_code_collection.original.LEK_UoEdinburghUK.SignalToConcentration import SI2Conc


# All tests will use the same arguments and same data...
arg_names = 'label', 'fa', 'tr', 'T1base', 'BLpts', 'r1', 's_array', 'conc_array', 'a_tol', 'r_tol'
test_data = SI2Conc_data.SI2Conc_data()


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels = [])
def test_LEK_UoEdinburghUK_SI2Conc(label, fa, tr, T1base, BLpts, r1, s_array, conc_array, a_tol, r_tol):

    #Prepare input data
    #Nothing to do for this function
    
    # run test
    conc_curve = SI2Conc.SI2Conc(s_array,tr,fa,T1base,[1,BLpts],S0=None)
    conc_array=conc_array*r1 # This function doesn't include r1, so multiply it out before testing
    np.testing.assert_allclose( [conc_curve], [conc_array], rtol=r_tol, atol=a_tol)

