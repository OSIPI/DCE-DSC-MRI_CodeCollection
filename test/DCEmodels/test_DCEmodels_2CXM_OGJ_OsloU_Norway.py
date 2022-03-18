import numpy as np

from ..helpers import osipi_parametrize
from . import DCEmodels_data
from src.original.OGJ_OsloU_Norway.MRImageAnalysis.DCE.Analyze import fitToModel

arg_names = 'label, t_array, C_t_array, cp_aif_array, vp_ref, ve_ref, fp_ref,'\
    'ps_ref, a_tol_vp, r_tol_vp, a_tol_ve, r_tol_ve, a_tol_fp, r_tol_fp,'\
    'a_tol_ps, r_tol_ps'
test_data = (DCEmodels_data.dce_DRO_data_2cxm())

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_OGJ_OsloU_Norway_2cxm_model(label, t_array, C_t_array,
                                      cp_aif_array, vp_ref, ve_ref, fp_ref,
                                      ps_ref, a_tol_vp, r_tol_vp, a_tol_ve,
                                      r_tol_ve, a_tol_fp, r_tol_fp, a_tol_ps,
                                      r_tol_ps):
    # NOTES:

    # prepare input data - create aif object
    t_array = t_array / 60

    # run test
    output = fitToModel('2CXM', C_t_array, t_array, cp_aif_array,
                        integrationMethod='trapezoidal', method='LLSQ',
                        showPbar=True)
    vp_meas = output.v_p
    ve_meas = output.v_e
    fp_meas = output.F_p * 100
    ps_meas = output.PS
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp,
                               atol=a_tol_vp)
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve,
                               atol=a_tol_ve)
    np.testing.assert_allclose([fp_meas], [fp_ref], rtol=r_tol_fp,
                               atol=a_tol_fp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps,
                               atol=a_tol_ps)
