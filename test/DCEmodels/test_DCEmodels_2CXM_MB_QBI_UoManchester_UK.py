import os
import numpy as np
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
from . import DCEmodels_data
from src.original.MB_QBI_UoManchester_UK.QbiPy.dce_models import two_cxm_model, \
    dce_aif

arg_names = 'label, t_array, C_t_array, cp_aif_array, vp_ref, ve_ref, fp_ref,' \
            'ps_ref, delay_ref, a_tol_vp, r_tol_vp, a_tol_ve, r_tol_ve, ' \
            'a_tol_fp, r_tol_fp, a_tol_ps, r_tol_ps, a_tol_delay, r_tol_delay'


filename_prefix = ''


def setup_module(module):
    # initialize the logfiles
    global filename_prefix # we want to change the global variable
    os.makedirs('./test/results/DCEmodels', exist_ok=True)
    filename_prefix = 'DCEmodels/TestResults_models'
    log_init(filename_prefix, '_MB_QBI_UoManchester_UK_2CXM', ['label', 'time (us)', 'vp_ref', 've_ref', 'fp_ref', 'ps_ref', 'vp_meas', 've_meas', 'fp_meas', 'ps_meas'])

test_data = (DCEmodels_data.dce_DRO_data_2cxm())
# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_MB_QBI_UoManchester_UK_2cxm_model(label, t_array, C_t_array,
                                          cp_aif_array, vp_ref, ve_ref,
                                          fp_ref, ps_ref, delay_ref,
                                          a_tol_vp, r_tol_vp, a_tol_ve,
                                          r_tol_ve, a_tol_fp, r_tol_fp,
                                          a_tol_ps, r_tol_ps, a_tol_delay,
                                          r_tol_delay):
    # NOTES:

    # prepare input data - create aif object
    t_array /= 60  # convert to minutes
    aif = dce_aif.Aif(times=t_array, base_aif=cp_aif_array,
                      aif_type=dce_aif.AifType(3))

    # run code
    tic = perf_counter()
    fp_meas, ps_meas, ve_meas, vp_meas = two_cxm_model.solve_LLS(C_t_array,
                                                                 aif, 0)
    fp_meas *= 100  # convert from ml/ml/min to ml/100/min
    exc_time = 1e6 * (perf_counter() - tic)  # measure execution time

    # log results
    log_results(filename_prefix, '_MB_QBI_UoManchester_UK_2CXM', [
        [label, f"{exc_time:.0f}", vp_ref, ve_ref, fp_ref, ps_ref, vp_meas, ve_meas, fp_meas, ps_meas]])

    # run test
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp,
                               atol=a_tol_vp)
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve,
                               atol=a_tol_ve)
    np.testing.assert_allclose([fp_meas], [fp_ref], rtol=r_tol_fp,
                               atol=a_tol_fp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps,
                               atol=a_tol_ps)
