import numpy as np
import os
from scipy.optimize import curve_fit
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
from . import DCEmodels_data
from src.original.LEK_UoEdinburgh_UK.PharmacokineticModelling.models import \
    Patlak

arg_names = 'label, t_array, C_t_array, cp_aif_array, vp_ref, ps_ref, ' \
            'delay_ref, a_tol_vp, r_tol_vp, a_tol_ps, r_tol_ps, a_tol_delay, ' \
            'r_tol_delay'


filename_prefix = ''


def setup_module(module):
    # initialize the logfiles
    global filename_prefix # we want to change the global variable
    os.makedirs('./test/results/DCEmodels', exist_ok=True)
    filename_prefix = 'DCEmodels/TestResults_models'
    log_init(filename_prefix, '_LEK_UoEdinburgh_UK_patlak', ['label', 'time (us)', 'vp_ref', 'ps_ref', 'delay_ref', 'vp_meas', 'ps_meas', 'delay_meas'])


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
test_data = (DCEmodels_data.dce_DRO_data_Patlak())
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_LEK_UoEdinburgh_UK_Patlak_model(label, t_array, C_t_array, cp_aif_array,
                                        vp_ref, ps_ref, delay_ref, a_tol_vp,
                                        r_tol_vp, a_tol_ps, r_tol_ps,
                                        a_tol_delay, r_tol_delay):
    # NOTES:
    # Fitting not implemented

    # prepare input data
    t_array = t_array / 60  # convert to mins, so that ps is in /min
    X0 = (0.6, 0.01)  # ps, vp starting values
    bounds = ((0.0, 0.0), (5.0, 1.0))

    # run code
    tic = perf_counter()
    output, pcov = curve_fit(lambda t, x, y: Patlak([x, y], t, cp_aif_array,
                                                    toff=0),
                             t_array, C_t_array, p0=X0,
                             bounds=bounds)
    ps_meas, vp_meas = output
    exc_time = 1e6 * (perf_counter() - tic)  # measure execution time

    # log results
    log_results(filename_prefix, '_LEK_UoEdinburgh_UK_patlak', [
        [label, f"{exc_time:.0f}", vp_ref, ps_ref, delay_ref, vp_meas, ps_meas, delay_ref]]) # in this case delay_ref is used as delay_meas was 0

    # run test
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp,
                               atol=a_tol_vp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps,
                               atol=a_tol_ps)


test_data_delay = (DCEmodels_data.dce_DRO_data_Patlak(delay=True))
# Use the test data to generate a parametrize decorator. This causes the
# following test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data_delay, xf_labels=[])
def test_LEK_UoEdinburgh_UK_Patlak_model_delay(label, t_array, C_t_array,
                                              cp_aif_array, vp_ref, ps_ref,
                                              delay_ref, a_tol_vp, r_tol_vp,
                                              a_tol_ps, r_tol_ps,
                                              a_tol_delay, r_tol_delay):
    # NOTES:
    # Fitting not implemented

    # prepare input data
    t_array = t_array / 60  # convert to mins, so that ps is in /min
    X0 = (0.6, 0.01, 0)  # ps, vp starting values
    bounds = ((0.0, 0.0, -10/60), (5.0, 1.0, 10/60))

    # run code
    tic = perf_counter()
    output, pcov = curve_fit(lambda t, x, y, delay: Patlak([x, y], t,
                                                           cp_aif_array,
                                                           toff=delay),
                             t_array, C_t_array, p0=X0,
                             bounds=bounds)
    ps_meas, vp_meas, delay_meas = output
    delay_meas *= 60
    exc_time = 1e6 * (perf_counter() - tic)  # measure execution time

    # log results
    log_results(filename_prefix, '_LEK_UoEdinburgh_UK_patlak', [
        [label, f"{exc_time:.0f}", vp_ref, ps_ref, delay_ref, vp_meas, ps_meas, delay_meas]])

    # run test
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp,
                               atol=a_tol_vp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps,
                               atol=a_tol_ps)
    np.testing.assert_allclose([delay_meas], [delay_ref], rtol=r_tol_delay,
                               atol=a_tol_delay)
