import numpy as np

from ..helpers import osipi_parametrize
from . import DCEmodels_data
from src.original.LCB_BNI.dce import fit_tofts

arg_names = 'label, t_array, C_array, ca_array, ta_array, ve_ref, Ktrans_ref, arterial_delay_ref, a_tol_ve, r_tol_ve, ' \
            'a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay '

test_data = (DCEmodels_data.dce_DRO_data_tofts())

# Use the test data to generate a parametrize decorator. This causes the following test to be run for every test case
# listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_LCB_BNI_tofts_model(label, t_array, C_array, ca_array, ta_array, ve_ref, Ktrans_ref, arterial_delay_ref,
                             a_tol_ve, r_tol_ve, a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay):
    # NOTES: delay fitting not implemented

    # prepare input data
    t_array = t_array  # /60  - in seconds

    # run test
    Ktrans_meas, ve_meas, Ct_fit = fit_tofts(t_array, C_array, ca_array)
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve, atol=a_tol_ve)
    np.testing.assert_allclose([Ktrans_meas], [Ktrans_ref], rtol=r_tol_Ktrans, atol=a_tol_Ktrans)
