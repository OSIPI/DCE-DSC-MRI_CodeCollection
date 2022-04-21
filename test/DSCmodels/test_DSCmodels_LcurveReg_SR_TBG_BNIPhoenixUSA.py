import os
import numpy as np
from time import perf_counter
from ..helpers import osipi_parametrize
from . import DSCmodels_data
from src.original.SR_TBG_BNIPhoenixUSA.AIFDeconvolution.AIF_deconvolution import AIFdeconvolution
from src.original.SR_TBG_BNIPhoenixUSA.DSCparameters.DSC_parameters import DSCparameters

# All tests will use the same arguments and same data...
arg_names = 'label, C_tis, C_aif, tr, cbv_ref, cbf_ref,' \
            'r_tol_cbv, r_tol_cbf, a_tol_cbv, a_tol_cbf'

test_data = DSCmodels_data.dsc_DRO_data_vascular_model()
filename_prefix = ''

# Use the test data to generate a parametrize decorator. This causes the following test to be run for every test case
# listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_SR_TBG_BNIPhoenixUSA_LcurveReg(label, C_tis, C_aif, tr, cbv_ref, cbf_ref,
                                        r_tol_cbv, r_tol_cbf, a_tol_cbv, a_tol_cbf):

    # Estimate perfusion scenario    
    residualFunction,_,_,_ = AIFdeconvolution(C_aif, C_tis, tr)
    cbf_meas,cbv_meas,_ = DSCparameters(C_tis, C_aif, residualFunction, tr)

    # test perfusion estimation    
    np.testing.assert_allclose(cbv_meas, cbv_ref, rtol=r_tol_cbv, atol=a_tol_cbv)
    np.testing.assert_allclose(cbf_meas, cbf_ref, rtol=r_tol_cbf, atol=a_tol_cbf)


