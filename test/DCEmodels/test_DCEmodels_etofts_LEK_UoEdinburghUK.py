import numpy as np
from scipy.optimize import curve_fit

from ..helpers import osipi_parametrize
from . import DCEmodels_data
from osipi_code_collection.original.LEK_UoEdinburghUK.PharmacokineticModelling.models import ExtKety

# All tests will use the same arguments and same data...
arg_names = 'label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref, a_tol_ve, ' \
            'r_tol_ve, a_tol_vp, r_tol_vp, a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay'

test_data = (
        DCEmodels_data.dce_DRO_data_extended_tofts_kety() +
        DCEmodels_data.dce_DRO_data_extended_tofts_kety(delay=True))

# Use the test data to generate a parametrize decorator. This causes the following test to be run for every test case
# listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_LEK_UoEdinburghUK_extended_tofts_kety_model(label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref,
                                                     Ktrans_ref, arterial_delay_ref, a_tol_ve, r_tol_ve, a_tol_vp,
                                                     r_tol_vp, a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay):
    # NOTES:
    # prepare input data
    t_array = t_array / 60
    arterial_delay_ref = arterial_delay_ref / 60
    X0 = (0.02, 0.2, 0.6, 0)
    bounds = ((0.0, 0.0, 0.0, 0), (0.7, 1, 5.0, 1))

    output, pcov = curve_fit(lambda t, x, y, z, toff: ExtKety([x, y, z], t, ca_array, toff), t_array, C_array, p0=X0,
                             bounds=bounds)

    Ktrans_meas, ve_meas, vp_meas, arterial_delay_meas = output

    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve, atol=a_tol_ve)
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp, atol=a_tol_vp)
    np.testing.assert_allclose([Ktrans_meas], [Ktrans_ref], rtol=r_tol_Ktrans, atol=a_tol_Ktrans)
    np.testing.assert_allclose([arterial_delay_meas], [arterial_delay_ref], rtol=r_tol_delay, atol=a_tol_delay)
