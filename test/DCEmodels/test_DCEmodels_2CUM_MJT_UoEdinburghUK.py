import numpy as np

from ..helpers import osipi_parametrize
from . import DCEmodels_data
from src.original.MJT_UoEdinburghUK import dce_fit, pk_models, aifs

arg_names = 'label, t_array, C_t_array, cp_aif_array, vp_ref, fp_ref,'\
    'ps_ref, a_tol_vp, r_tol_vp, a_tol_fp, r_tol_fp,'\
    'a_tol_ps, r_tol_ps'
test_data = (DCEmodels_data.dce_DRO_data_2cum())

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_MJT_UoEdinburghUK_2cum_model(label, t_array, C_t_array,
                                      cp_aif_array, vp_ref, fp_ref,
                                      ps_ref, a_tol_vp, r_tol_vp, a_tol_fp,
                                      r_tol_fp, a_tol_ps, r_tol_ps):
    # NOTES:

    # prepare input data - create aif object
    aif = aifs.patient_specific(t_array, cp_aif_array)
    pk_model = pk_models.tcum(t_array, aif)

    # run test
    pk_pars, C_t_fit = dce_fit.conc_to_pkp(C_t_array, pk_model)
    vp_meas = pk_pars['vp']
    fp_meas = pk_pars['fp']
    ps_meas = pk_pars['ps']
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp,
                               atol=a_tol_vp)
    np.testing.assert_allclose([fp_meas], [fp_ref], rtol=r_tol_fp,
                               atol=a_tol_fp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps,
                               atol=a_tol_ps)
