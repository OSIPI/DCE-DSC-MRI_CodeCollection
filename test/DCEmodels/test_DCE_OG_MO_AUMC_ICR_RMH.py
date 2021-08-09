import pytest
import numpy as np

from ..helpers import osipi_parametrize
from . import dce_data
from src.original.OG_MO_AUMC_ICR_RMH.ExtendedTofts.DCE import fit_tofts_model, fit_aif


# All tests will use the same arguments and same data...
arg_names = 'label, t_array, C_array, ca_array, ta_ref, ve_ref, vp_ref, Ktrans_ref, a_tol, r_tol'
test_data = (
    dce_data.dce_test_data()
    )


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
# In the following test, we specify 1 case that is expected to fail...
@osipi_parametrize(arg_names, test_data)
def testOG_MO_AUMC_ICR_RMH_tofts_model(label, t_array, C_array, ca_array, ta_ref, ve_ref, vp_ref, Ktrans_ref, a_tol, r_tol):
    # NOTES:

    # prepare input data
    AIF=fit_aif(ca_array, t_array, model='Cosine8')
    C_array=np.array(C_array)
    C_array=C_array[np.newaxis,...]
    # run test
    ke, ta_meas, ve_meas, vp_meas = fit_tofts_model(C_array, t_array, AIF, idxs=None, X0=(0.6, 1, 0.03, 0.0025), bounds=((0.0, 0, 0.0, 0.0), (5.0, 2, 1.0, 1.0)),
                    jobs=4, model='Cosine8')

    Ktrans_meas = ke * ve_meas

    np.testing.assert_allclose( [ve_meas], [ve_ref], rtol=r_tol, atol=a_tol )
    np.testing.assert_allclose( [vp_meas], [vp_ref], rtol=r_tol, atol=a_tol )
    np.testing.assert_allclose( [Ktrans_meas], [Ktrans_ref], rtol=r_tol, atol=a_tol )
    np.testing.assert_allclose( [ta_meas], [ta_ref], rtol=r_tol, atol=a_tol )
