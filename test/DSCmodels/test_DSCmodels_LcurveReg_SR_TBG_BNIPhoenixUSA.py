import os
import numpy as np
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
from . import DSCmodels_data
from src.original.SR_TBG_BNI_USAPhoenixUSA.AIFDeconvolution.AIF_deconvolution import AIFdeconvolution
from src.original.SR_TBG_BNI_USAPhoenixUSA.DSCparameters.DSC_parameters import DSCparameters

# All tests will use the same arguments and same data...
arg_names = 'label, C_tis, C_aif, tr, cbv_ref, cbf_ref,' \
            'r_tol_cbv, r_tol_cbf, a_tol_cbv, a_tol_cbf'

test_data = DSCmodels_data.dsc_DRO_data_vascular_model()

filename_prefix = ''


def setup_module(module):
    # initialize the logfiles
    global filename_prefix # we want to change the global variable
    os.makedirs('./test/results/DSCmodels', exist_ok=True)
    filename_prefix = 'DSCmodels/TestResults_ParamEstimation'
    log_init(filename_prefix, '_SR_TBG_BNI_USAPhoenix_USA', ['label', 'time (us)', 'cbv_ref', 'cbf_ref', 'cbv_meas', 'cbf_meas'])

# Use the test data to generate a parametrize decorator. This causes the following test to be run for every test case
# listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_SR_TBG_BNI_USAPhoenixUSA_LcurveReg(label, C_tis, C_aif, tr, cbv_ref, cbf_ref,
                                        r_tol_cbv, r_tol_cbf, a_tol_cbv, a_tol_cbf):

    # run code
    tic = perf_counter()
    residualFunction,_,_,_ = AIFdeconvolution(C_aif, C_tis, tr)
    cbf_meas,cbv_meas,_ = DSCparameters(C_tis, C_aif, residualFunction, tr)
    exc_time = 1e6 * (perf_counter() - tic)  # measure execution time

    # log results
    log_results(filename_prefix, '_SR_TBG_BNI_USAPhoenix_USA', [
        [label, f"{exc_time:.0f}", cbv_ref, cbf_ref, cbv_meas, cbf_meas]])

    # run test
    np.testing.assert_allclose(cbv_meas, cbv_ref, rtol=r_tol_cbv, atol=a_tol_cbv)
    np.testing.assert_allclose(cbf_meas, cbf_ref, rtol=r_tol_cbf, atol=a_tol_cbf)


