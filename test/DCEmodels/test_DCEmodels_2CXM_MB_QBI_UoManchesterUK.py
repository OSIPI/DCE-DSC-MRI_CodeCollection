import numpy as np

from ..helpers import osipi_parametrize
from . import DCEmodels_data
from src.original.MB_QBI_UoManchesterUK.QbiPy.dce_models import two_cxm_model, \
    dce_aif


arg_names = 'label, t_array, C_t_array, cp_aif_array, vp_ref, ve_ref, fp_ref,'\
    'ps_ref, a_tol_vp, r_tol_vp, a_tol_ve, r_tol_ve, a_tol_fp, r_tol_fp,'\
    'a_tol_ps, r_tol_ps'
test_data = (DCEmodels_data.dce_DRO_data_2cxm())

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_MB_QBI_UoManchesterUK_2cxm_model(label, t_array,
                                                       C_t_array,
                                      cp_aif_array, vp_ref, ve_ref, fp_ref,
                                      ps_ref, a_tol_vp, r_tol_vp, a_tol_ve,
                                      r_tol_ve, a_tol_fp, r_tol_fp, a_tol_ps,
                                      r_tol_ps):
    # NOTES:

    # prepare input data - create aif object
    t_array /= 60  # convert to minutes
    aif = dce_aif.Aif(times=t_array, base_aif=cp_aif_array,
              aif_type=dce_aif.AifType(3))

    # run test
    fp_meas, ps_meas, ve_meas, vp_meas = two_cxm_model.solve_LLS(C_t_array,
                                                                 aif, 0)
    fp_meas *= 100  # convert from ml/ml/min to ml/100/min
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp,
                               atol=a_tol_vp)
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve,
                               atol=a_tol_ve)
    np.testing.assert_allclose([fp_meas], [fp_ref], rtol=r_tol_fp,
                               atol=a_tol_fp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps,
                               atol=a_tol_ps)
