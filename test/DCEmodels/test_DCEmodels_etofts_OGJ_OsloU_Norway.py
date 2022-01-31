import numpy as np

from ..helpers import osipi_parametrize
from . import DCEmodels_data
from src.original.OGJ_OsloU_Norway.MRImageAnalysis.DCE.Analyze import fitToModel

# All tests will use the same arguments and same data...
arg_names = 'label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref,  a_tol_ve, ' \
            'r_tol_ve, a_tol_vp,r_tol_vp,a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay '
test_data = (DCEmodels_data.dce_DRO_data_extended_tofts_kety())

# Use the test data to generate a parametrize decorator. This causes the following test to be run for every test case
# listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_OGJ_OsloU_Norway_extended_tofts_kety_model(label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref,
                                                    Ktrans_ref, arterial_delay_ref, a_tol_ve, r_tol_ve, a_tol_vp,
                                                    r_tol_vp, a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay):
    # NOTES: delay fitting not implemented

    # prepare input data
    t_array = t_array / 60

    # run test
    output = fitToModel('ETM', C_array, t_array, ca_array, integrationMethod='trapezoidal', method='LLSQ',
                        showPbar=True)
    Ktrans_meas = output.K_trans
    vp_meas = output.v_p
    ve_meas = output.v_e
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve, atol=a_tol_ve)
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp, atol=a_tol_vp)
    np.testing.assert_allclose([Ktrans_meas], [Ktrans_ref], rtol=r_tol_Ktrans, atol=a_tol_Ktrans)
