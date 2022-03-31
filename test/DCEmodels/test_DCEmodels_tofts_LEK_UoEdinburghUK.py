import os
import numpy as np
from scipy.optimize import curve_fit
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
from . import DCEmodels_data
from osipi_code_collection.original.LEK_UoEdinburghUK.PharmacokineticModelling.models import ExtKety, Kety

arg_names = 'label, t_array, C_array, ca_array, ta_array, ve_ref, Ktrans_ref, arterial_delay_ref,  a_tol_ve, ' \
            'r_tol_ve, a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay'
test_data = (DCEmodels_data.dce_DRO_data_tofts() +
             DCEmodels_data.dce_DRO_data_tofts(delay=True))

filename_prefix = ''

def setup_module(module):
    # initialize the logfiles
    global filename_prefix # we want to change the global variable
    os.makedirs('./test/results/DCEmodels', exist_ok=True)
    filename_prefix = 'DCEmodels/TestResults_models'
    log_init(filename_prefix, '_LEK_UoEdinburghUK_tofts_model', ['label', 'time (us)', 'Ktrans_ref', 've_ref', 'delay_ref', 'Ktrans_meas', 've_meas', 'delay_meas'])


# Use the test data to generate a parametrize decorator. This causes the following test to be run for every test case
# listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_LEK_UoEdinburghUK_tofts_model(label, t_array, C_array, ca_array, ta_array, ve_ref, Ktrans_ref,
                                       arterial_delay_ref, a_tol_ve, r_tol_ve, a_tol_Ktrans, r_tol_Ktrans, a_tol_delay,
                                       r_tol_delay):
    # NOTES:
    # Fitting not implemented

    # prepare input data
    t_array = t_array / 60  # convert to minutes so that KTrans is in /min
    X0 = (0.6, 0.2, 0)  # KTrans, ve, delay
    bounds = ((0.0, 0.0, -10/60), (5, 1, 10/60))

    # run code
    tic = perf_counter()
    output, pcov = curve_fit(lambda t, x, y, delay: Kety([x, y], t, ca_array, delay), t_array, C_array, p0=X0,
                             bounds=bounds)
    Ktrans_meas, ve_meas, arterial_delay_meas = output
    exc_time = 1e6 * (perf_counter() - tic)  # measure execution time

    # log results
    log_results(filename_prefix, '_LEK_UoEdinburghUK_tofts_model', [
        [label, f"{exc_time:.0f}", Ktrans_ref, ve_ref, arterial_delay_ref, Ktrans_meas, ve_meas, arterial_delay_meas]])

    # run test
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve, atol=a_tol_ve)
    np.testing.assert_allclose([Ktrans_meas], [Ktrans_ref], rtol=r_tol_Ktrans, atol=a_tol_Ktrans)
    np.testing.assert_allclose([arterial_delay_meas], [arterial_delay_ref], rtol=r_tol_delay, atol=a_tol_delay)
