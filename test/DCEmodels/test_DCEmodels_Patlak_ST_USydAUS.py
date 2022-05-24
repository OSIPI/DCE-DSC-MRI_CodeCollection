import os
import numpy as np
from scipy.optimize import curve_fit
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
from . import DCEmodels_data
from src.original.ST_USydAUS.ModelDictionary import PatlakModel

arg_names = 'label, t_array, C_t_array, cp_aif_array, vp_ref, ps_ref, ' \
            'delay_ref, a_tol_vp, r_tol_vp, a_tol_ps, r_tol_ps, a_tol_delay, ' \
            'r_tol_delay'
test_data = (DCEmodels_data.dce_DRO_data_Patlak())

filename_prefix = ''

def setup_module(module):
    # initialize the logfiles
    global filename_prefix # we want to change the global variable
    os.makedirs('./test/results/DCEmodels', exist_ok=True)
    filename_prefix = 'DCEmodels/TestResults_models'
    log_init(filename_prefix, '_ST_USydAus_patlak', ['label', 'time (us)', 'vp_ref', 'ps_ref', 'vp_meas', 'ps_meas'])

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_ST_USydAUS_Patlak_model(label, t_array, C_t_array,
                                              cp_aif_array, vp_ref, ps_ref,
                                              delay_ref, a_tol_vp, r_tol_vp,
                                              a_tol_ps, r_tol_ps,
                                              a_tol_delay, r_tol_delay):
    # NOTES:
    # Fitting not implemented
    # Delay not implemented

    # prepare input data
    t_array = t_array / 60  # convert to mins, so that ps is in /min
    data = np.column_stack((t_array, cp_aif_array))
    X0 = (0.01, 0.6)  # vp, ps starting values
    bounds = ((0.0, 0.0), (1.0, 5.0))

    # run code
    tic = perf_counter()
    output, pcov = curve_fit(PatlakModel, data, C_t_array, p0=X0, bounds=bounds)
    vp_meas, ps_meas = output
    exc_time = 1e6 * (perf_counter() - tic)  # measure execution time

    # log results
    log_results(filename_prefix, '_ST_USydAus_patlak', [
        [label, f"{exc_time:.0f}", vp_ref, ps_ref, vp_meas, ps_meas]])

    # run test
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp,
                               atol=a_tol_vp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps,
                               atol=a_tol_ps)
