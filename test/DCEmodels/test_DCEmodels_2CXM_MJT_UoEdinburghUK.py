import os
import numpy as np
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
from . import DCEmodels_data
from src.original.MJT_UoEdinburghUK import dce_fit, pk_models, aifs

arg_names = 'label, t_array, C_t_array, cp_aif_array, vp_ref, ve_ref, fp_ref,' \
            'ps_ref, delay_ref, a_tol_vp, r_tol_vp, a_tol_ve, r_tol_ve, ' \
            'a_tol_fp, r_tol_fp, a_tol_ps, r_tol_ps, a_tol_delay, r_tol_delay'


filename_prefix = ''


def setup_module(module):
    # initialize the logfiles
    global filename_prefix  # we want to change the global variable
    os.makedirs('./test/results/DCEmodels', exist_ok=True)
    filename_prefix = 'DCEmodels/TestResults_models'
    log_init(filename_prefix, '_MJT_UoEdinburghUK_2CXM',
             ['label', 'time (us)', 'vp_ref', 've_ref', 'fp_ref', 'ps_ref','delay_ref', 'vp_meas', 've_meas', 'fp_meas', 'ps_meas','delay_meas'])

test_data = (DCEmodels_data.dce_DRO_data_2cxm())
# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_MJT_UoEdinburghUK_2cxm_model(label, t_array, C_t_array,
                                      cp_aif_array, vp_ref, ve_ref,
                                      fp_ref, ps_ref, delay_ref,
                                      a_tol_vp, r_tol_vp, a_tol_ve,
                                      r_tol_ve, a_tol_fp, r_tol_fp,
                                      a_tol_ps, r_tol_ps, a_tol_delay,
                                      r_tol_delay):
    # NOTES:

    # prepare input data - create aif object
    aif = aifs.PatientSpecific(t_array, cp_aif_array)
    pk_model = pk_models.TCXM(t_array, aif, upsample_factor=3)

    # run code
    tic = perf_counter()
    vp_meas, ps_meas, ve_meas, fp_meas, C_t_fit = dce_fit.ConcToPKP(pk_model).proc(C_t_array)
    exc_time = 1e6 * (perf_counter() - tic)  # measure execution time

    # log results
    log_results(filename_prefix, '_MJT_UoEdinburghUK_2CXM', [
        [label, f"{exc_time:.0f}", vp_ref, ve_ref, fp_ref, ps_ref, delay_ref, vp_meas, ve_meas, fp_meas, ps_meas, delay_ref]])

    # run test
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp,
                               atol=a_tol_vp)
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve,
                               atol=a_tol_ve)
    np.testing.assert_allclose([fp_meas], [fp_ref], rtol=r_tol_fp,
                               atol=a_tol_fp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps,
                               atol=a_tol_ps)


test_data = (DCEmodels_data.dce_DRO_data_2cxm(delay=True))
# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_MJT_UoEdinburghUK_2cxm_model_delay(label, t_array, C_t_array,
                                      cp_aif_array, vp_ref, ve_ref,
                                      fp_ref, ps_ref, delay_ref,
                                      a_tol_vp, r_tol_vp, a_tol_ve,
                                      r_tol_ve, a_tol_fp, r_tol_fp,
                                      a_tol_ps, r_tol_ps, a_tol_delay,
                                      r_tol_delay):
    # NOTES:

    # prepare input data - create aif object
    aif = aifs.PatientSpecific(t_array, cp_aif_array)
    pk_model = pk_models.TCXM(t_array, aif, upsample_factor=3, fixed_delay=None)

    # run code
    tic = perf_counter()
    vp_meas, ps_meas, ve_meas, fp_meas, delay_meas, C_t_fit = dce_fit.ConcToPKP(pk_model).proc(C_t_array)
    exc_time = 1e6 * (perf_counter() - tic)  # measure execution time

    # log results
    log_results(filename_prefix, '_MJT_UoEdinburghUK_2CXM', [
        [label, f"{exc_time:.0f}", vp_ref, ve_ref, fp_ref, ps_ref, delay_ref, vp_meas, ve_meas, fp_meas, ps_meas, delay_meas]])

    # run test
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp,
                               atol=a_tol_vp)
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve,
                               atol=a_tol_ve)
    np.testing.assert_allclose([fp_meas], [fp_ref], rtol=r_tol_fp,
                               atol=a_tol_fp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps,
                               atol=a_tol_ps)
    np.testing.assert_allclose([delay_meas], [delay_ref], rtol=r_tol_delay,
                               atol=a_tol_delay)
