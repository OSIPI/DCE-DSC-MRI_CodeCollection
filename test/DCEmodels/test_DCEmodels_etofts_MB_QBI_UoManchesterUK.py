import os
import numpy as np
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
from . import DCEmodels_data
from osipi_dce_dsc_repo.original.MB_QBI_UoManchesterUK.QbiPy.dce_models import tofts_model, dce_aif

# All tests will use the same arguments and same data...
arg_names = 'label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref,  a_tol_ve, ' \
            'r_tol_ve, a_tol_vp,r_tol_vp,a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay '
test_data = (DCEmodels_data.dce_DRO_data_extended_tofts_kety())

filename_prefix = ''

def setup_module(module):
    # initialize the logfiles
    global filename_prefix # we want to change the global variable
    os.makedirs('./test/results/DCEmodels', exist_ok=True)
    filename_prefix = 'DCEmodels/TestResults_models'
    log_init(filename_prefix, '_MB_QBI_UoManchester_etofts', ['label', 'time (us)', 'Ktrans_ref', 've_ref', 'vp_ref', 'Ktrans_meas', 've_meas', 'vp_meas'])


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_MB_QBI_UoManchester_extended_tofts_kety_model(label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref,
                                                       Ktrans_ref, arterial_delay_ref, a_tol_ve, r_tol_ve, a_tol_vp,
                                                       r_tol_vp, a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay):
    # NOTES:
    # Artery-capillary delay fitting not implemented

    # prepare input data
    t_array = t_array / 60  # convert to minutes
    aif = dce_aif.Aif(times=t_array, base_aif=ca_array, aif_type=dce_aif.AifType(3))

    # run code
    tic = perf_counter()
    Ktrans_meas, ve_meas, vp_meas = tofts_model.solve_LLS(C_array, aif, 0)
    exc_time = 1e6 * (perf_counter() - tic)  # measure execution time

    # log results
    log_results(filename_prefix, '_MB_QBI_UoManchester_etofts', [[label, f"{exc_time:.0f}", Ktrans_ref, ve_ref, vp_ref, Ktrans_meas, ve_meas, vp_meas]])

    # run test
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve, atol=a_tol_ve)
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp, atol=a_tol_vp)
    np.testing.assert_allclose([Ktrans_meas], [Ktrans_ref], rtol=r_tol_Ktrans, atol=a_tol_Ktrans)
