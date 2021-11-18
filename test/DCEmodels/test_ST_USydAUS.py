import pytest
import numpy as np

from ..helpers import osipi_parametrize
from . import dce_data
import matplotlib.pyplot as plt
from src.original.ST_USydAUS.ModelDictionary import ExtendedTofts, Tofts
from scipy.optimize import curve_fit
from .tools import append_to_excel
import inspect

# All tests will use the same arguments and same data...
arg_names = 'label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref,  a_tol_ve, r_tol_ve, a_tol_vp,r_tol_vp,a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay'
test_data = (
    dce_data.dce_DRO_data()
    )


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
# In the following test, we specify 1 case that is expected to fail...
@osipi_parametrize(arg_names, test_data, xf_labels = ['test_vox_WM', 'test_vox_GM','test_vox_WM_10','test_vox_GM_10','test_vox_WM_20','test_vox_GM_20','test_vox_WM_30','test_vox_GM_30','test_vox_WM_50','test_vox_GM_50'])
def testST_USydAUS_ext_tofts_model(label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref,a_tol_ve, r_tol_ve, a_tol_vp,r_tol_vp,a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay):
    # NOTES:

    # prepare input data
    ta_array = ta_array/60
    data=np.column_stack((ta_array,ca_array))
    X0 = (0.6, 0.2, 0.02)
    bounds = ((0.0, 0.0, 0.0), (5.0, 1, 0.7))
    output, pcov = curve_fit(ExtendedTofts, data, C_array, p0=X0, bounds=bounds)

    vp_meas, ve_meas, Ktrans_meas = output

    print(['ve meas vs ref '+ str(ve_meas)+' vs '+str(ve_ref)])
    print(['vp meas vs ref '+ str(vp_meas) + ' vs ' +str(vp_ref)])
    print(['Kt meas vs ref '+ str(Ktrans_meas) + ' vs ' + str(Ktrans_ref)])

    data = [[inspect.stack()[0][3],label,ve_ref,vp_ref,Ktrans_ref,ve_meas,vp_meas,Ktrans_meas]]
    columns = ['testname','label','ve_ref','vp_ref','ktrans_ref','ve_meas','vp_meas','ktrans_meas']
    append_to_excel(data, columns)
    
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve, atol=a_tol_ve)
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp, atol=a_tol_vp)
    np.testing.assert_allclose([Ktrans_meas], [Ktrans_ref], rtol=r_tol_Ktrans, atol=a_tol_Ktrans)

arg_names = 'label, t_array, C_array, ca_array, ta_array, ve_ref, Ktrans_ref, arterial_delay_ref,  a_tol_ve, r_tol_ve, a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay'
test_data = (
    dce_data.dce_DRO_data_tofts()
    )


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
# In the following test, we specify 1 case that is expected to fail...
@osipi_parametrize(arg_names, test_data, xf_labels = ['test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii',
                                                      'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii20',
                                                      'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii30',
                                                      'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii50',
                                                      'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii100',
                                                      #   'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii_delayed',
                                                      # 'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii20_delayed',
                                                      # 'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii30_delayed',
                                                      # 'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii50_delayed',
                                                      # 'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii100_delayed',
                                                      #'test_vox_VIF_6s_jit_0s_T1_500_S0_500_sigma_5.nii.gz',
                                                      #'test_vox_VIF_6s_jit_0s_T1_500_S0_1000_sigma_10.nii.gz',
                                                      #'test_vox_VIF_6s_jit_0s_T1_500_S0_500_sigma_50.nii.gz',
                                                      #'test_vox_VIF_6s_jit_3s_T1_500_S0_500_sigma_5.nii.gz',
                                                      #'test_vox_T1_6s_jit_3s_T1_500_S0_500_sigma_5.nii.gz',
                                                      #'test_vox_T2_6s_jit_3s_T1_500_S0_500_sigma_5.nii.gz',
                                                      #'test_vox_T3_6s_jit_3s_T1_500_S0_500_sigma_5.nii.gz',
                                                      #'test_vox_T4_6s_jit_3s_T1_500_S0_500_sigma_5.nii.gz',
                                                      #'test_vox_T5_6s_jit_3s_T1_500_S0_500_sigma_5.nii.gz',
                                                      #'test_vox_VIF_6s_jit_3s_T1_500_S0_1000_sigma_10.nii.gz',
                                                      #'test_vox_T1_6s_jit_3s_T1_500_S0_1000_sigma_10.nii.gz',
                                                      #'test_vox_T2_6s_jit_3s_T1_500_S0_1000_sigma_10.nii.gz',
                                                      #'test_vox_T3_6s_jit_3s_T1_500_S0_1000_sigma_10.nii.gz',
                                                      #'test_vox_T4_6s_jit_3s_T1_500_S0_1000_sigma_10.nii.gz',
                                                      #'test_vox_T5_6s_jit_3s_T1_500_S0_1000_sigma_10.nii.gz',
                                                      #'test_vox_VIF_6s_jit_3s_T1_500_S0_500_sigma_50.nii.gz',
                                                      #'test_vox_T1_6s_jit_3s_T1_500_S0_500_sigma_50.nii.gz',
                                                      #'test_vox_T2_6s_jit_3s_T1_500_S0_500_sigma_50.nii.gz',
                                                      #'test_vox_T3_6s_jit_3s_T1_500_S0_500_sigma_50.nii.gz',
                                                      #'test_vox_T4_6s_jit_3s_T1_500_S0_500_sigma_50.nii.gz',
                                                      #'test_vox_T5_6s_jit_3s_T1_500_S0_500_sigma_50.nii.gz'
                                                      ])

def testST_USydAUS_tofts_model(label, t_array, C_array, ca_array, ta_array, ve_ref, Ktrans_ref, arterial_delay_ref,a_tol_ve, r_tol_ve, a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay):
    # NOTES:

    # prepare input data
    ta_array = ta_array/60
    data=np.column_stack((ta_array,ca_array))
    X0 = (0.2, 0.6)
    bounds = ((0.0, 0.0), (1, 5.0))
    
    output, pcov = curve_fit(Tofts, data, C_array, p0=X0, bounds=bounds)

    ve_meas, Ktrans_meas = output

    print(['ve meas vs ref '+ str(ve_meas)+' vs '+str(ve_ref)])
    print(['Kt meas vs ref '+ str(Ktrans_meas) + ' vs ' + str(Ktrans_ref)])

    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve, atol=a_tol_ve)
    np.testing.assert_allclose([Ktrans_meas], [Ktrans_ref], rtol=r_tol_Ktrans, atol=a_tol_Ktrans)