import pytest
import numpy as np

from ..helpers import osipi_parametrize
from . import DCEmodels_data
from src.original.MB_QBI_UoManchesterUK.QbiPy.dce_models import tofts_model, dce_aif


# All tests will use the same arguments and same data...
arg_names = 'label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref,  a_tol_ve, r_tol_ve, a_tol_vp,r_tol_vp,a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay'
test_data = (
    DCEmodels_data.dce_DRO_data_extended_tofts_kety()
)


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
# In the following test, we specify 1 case that is expected to fail...
@osipi_parametrize(arg_names, test_data,
                   xf_labels=['test_vox_VIF','test_vox_VIF_30','test_vox_VIF_50','test_vox_VIF_100','test_vox_WM', 'test_vox_GM', 'test_vox_WM_10', 'test_vox_GM_10', 'test_vox_WM_20',
                              'test_vox_GM_20', 'test_vox_WM_30', 'test_vox_GM_30', 'test_vox_WM_50', 'test_vox_GM_50'])
def test_MB_QBI_UoManchester_extended_tofts_kety_model(label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref,
                                       arterial_delay_ref, a_tol_ve, r_tol_ve, a_tol_vp, r_tol_vp, a_tol_Ktrans,
                                       r_tol_Ktrans, a_tol_delay, r_tol_delay):
    # NOTES:

    # prepare input data - create aif object
    t_array = t_array /60#  - in seconds
    aif = dce_aif.Aif(times=t_array, base_aif=ca_array, aif_type =  dce_aif.AifType(3))
    #aif.base_aif_ = ca_array
    #aif.type_ = dce_aif.AifType(3)
    Ktrans_meas, ve_meas, vp_meas = tofts_model.solve_LLS(
        C_array, aif, 0)

    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve, atol=a_tol_ve)
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp, atol=a_tol_vp)
    np.testing.assert_allclose([Ktrans_meas], [Ktrans_ref], rtol=r_tol_Ktrans, atol=a_tol_Ktrans)
