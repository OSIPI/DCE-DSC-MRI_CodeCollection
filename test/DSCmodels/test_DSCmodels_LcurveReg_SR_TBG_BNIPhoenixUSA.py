import os
import numpy as np

from ..helpers import osipi_parametrize
from . import DSCmodels_data
from src.original.SR_TBG_BNIPhoenixUSA.AIFDeconvolution.AIF_deconvolution import AIFdeconvolution
from src.original.SR_TBG_BNIPhoenixUSA.DSCparameters.DSC_parameters import DSCparameters

# All tests will use the same arguments and same data...
arg_names = 'label, C_gray, C_white, C_aif, tr, cbv_g, cbv_w, cbf_g, cbf_w,' \
            'r_tol_cbv_g, r_tol_cbv_w, r_tol_cbf_g, r_tol_cbf_w,' \
            'a_tol_cbv_g, a_tol_cbv_w, a_tol_cbf_g, a_tol_cbf_w'

test_data = DSCmodels_data.dsc_DRO_data_vascular_model()
filename_prefix = ''


# Use the test data to generate a parametrize decorator. This causes the following test to be run for every test case
# listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_SR_TBG_BNIPhoenixUSA_LcurveReg(label, C_gray, C_white, C_aif, tr, cbv_g, cbv_w, cbf_g, cbf_w,
                                        r_tol_cbv_g, r_tol_cbv_w, r_tol_cbf_g, r_tol_cbf_w,
                                        a_tol_cbv_g, a_tol_cbv_w, a_tol_cbf_g, a_tol_cbf_w):
    # run code
    cbf_g_meas = [0] * len(cbf_g)
    cbv_g_meas = [0] * len(cbf_g)
    cbf_w_meas = [0] * len(cbf_w)
    cbv_w_meas = [0] * len(cbf_w)

    # Estimate perfusion scenarios (only one curve per simulated scenario.)
    for i in range(len(cbf_g)): 
        C_tis = C_gray[str(cbf_g[i])][0,:]
        residualFunction,_,_,_ = AIFdeconvolution(C_aif, C_tis, tr)
        cbf_g_meas[i],cbv_g_meas[i],_= DSCparameters(C_tis, C_aif, residualFunction, tr)
    
    for i in range(len(cbf_w)): # Perfusion scenarios, only evaluating one curve per simulated scenario.    
        C_tis = C_white[str(cbf_w[i])][0,:]
        residualFunction,_,_,_ = AIFdeconvolution(C_aif, C_tis, tr)
        cbf_w_meas[i],cbv_w_meas[i],_= DSCparameters(C_tis, C_aif, residualFunction, tr)

    # Run tests for each simulated perfusion scenario (cbv and cbf)
    for i in range(len(cbf_g)):
        np.testing.assert_allclose(cbv_g_meas[i], cbv_g*100, rtol=r_tol_cbv_g, atol=a_tol_cbv_g)
        np.testing.assert_allclose(cbv_w_meas[i], cbv_w*100, rtol=r_tol_cbv_w, atol=a_tol_cbv_w)

    for i in range(len(cbf_w)):
        np.testing.assert_allclose(cbf_g_meas[i], cbf_g[i], rtol=r_tol_cbf_g, atol=a_tol_cbf_g)
        np.testing.assert_allclose(cbf_w_meas[i], cbf_w[i], rtol=r_tol_cbf_w, atol=a_tol_cbf_w)


