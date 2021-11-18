import pytest
import numpy as np

from ..helpers import osipi_parametrize
from . import dce_data
import matplotlib.pyplot as plt
from src.original.OGJ_OsloU_Norway.MRImageAnalysis.DCE.Analyze import fitToModel
from scipy.optimize import curve_fit
import inspect


# All tests will use the same arguments and same data...
arg_names = 'label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref,  a_tol_ve, r_tol_ve, a_tol_vp,r_tol_vp,a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay'
test_data = (
    dce_data.dce_DRO_data()
    )


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
# In the following test, we specify 1 case that is expected to fail...
@osipi_parametrize(arg_names, test_data, xf_labels = ['test_vox_VIF','test_vox_VIF_30','test_vox_VIF_50','test_vox_VIF_100','test_vox_WM', 'test_vox_GM','test_vox_WM_10','test_vox_GM_10','test_vox_WM_20','test_vox_GM_20','test_vox_WM_30','test_vox_GM_30','test_vox_WM_50','test_vox_GM_50'])
def test_OGJ_OsloU_Norway_tofts_model(label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref,a_tol_ve, r_tol_ve, a_tol_vp,r_tol_vp,a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay):
    # NOTES:

    # prepare input data
    t_array = t_array/60
    
    # X0 = (0.02, 0.02, 0.2, 0.6)
    # bounds = ((0.0, 0.0, 0.0, 0.0), (0.7, 0.7, 1, 5.0))
    # output, pcov = curve_fit(lambda t,x,y,z,w: ETM(t_array,ca_array,x,y,z,w), t_array, C_array, p0=   X0, bounds=bounds)
    
    output = fitToModel('ETM', C_array, t_array, ca_array, integrationMethod='trapezoidal', method='LLSQ', showPbar=True)
    Ktrans_meas = output.K_trans
    vp_meas = output.v_p
    ve_meas = output.v_e
    
    print(['ve meas vs ref '+ str(ve_meas)+' vs '+str(ve_ref)])
    print(['vp meas vs ref '+ str(vp_meas) + ' vs ' +str(vp_ref)])
    print(['Kt meas vs ref '+ str(Ktrans_meas) + ' vs ' + str(Ktrans_ref)])

    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve, atol=a_tol_ve)
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp, atol=a_tol_vp)
    np.testing.assert_allclose([Ktrans_meas], [Ktrans_ref], rtol=r_tol_Ktrans, atol=a_tol_Ktrans)


arg_names = 'label, t_array, C_array, ca_array, ta_array, ve_ref, Ktrans_ref, arterial_delay_ref,  a_tol_ve, r_tol_ve, a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay'
test_data = (
    dce_data.dce_DRO_data_tofts() +
    dce_data.dce_DRO_data_tofts(delay=True)
    )


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
# In the following test, we specify 1 case that is expected to fail...
@osipi_parametrize(arg_names, test_data, xf_labels = ['test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii',
                                                      'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii20',
                                                      'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii30',
                                                      'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii50',
                                                      'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii100',
                                                        'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii_delayed',
                                                      'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii20_delayed',
                                                      'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii30_delayed',
                                                      'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii50_delayed',
                                                      'test_vox_VIF_T1_500_noiseless/20150428_085000dynamics016a001.nii100_delayed',
                                                      ])
def test_LEK_UoEdinburghUK_tofts_model(label, t_array, C_array, ca_array, ta_array, ve_ref, Ktrans_ref, arterial_delay_ref,a_tol_ve, r_tol_ve, a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay):
    # NOTES:

    # prepare input data
    # prepare input data
    t_array = t_array / 60

    # X0 = (0.02, 0.02, 0.2, 0.6)
    # bounds = ((0.0, 0.0, 0.0, 0.0), (0.7, 0.7, 1, 5.0))
    # output, pcov = curve_fit(lambda t,x,y,z,w: ETM(t_array,ca_array,x,y,z,w), t_array, C_array, p0=   X0, bounds=bounds)

    output = fitToModel('TM', C_array, t_array, ca_array, integrationMethod='trapezoidal', method='LLSQ',
                        showPbar=True)
    Ktrans_meas = output.K_trans
    ve_meas = output.v_e

    #plt.plot(t_array, Kety([Ktrans_meas, ve_meas], t_array, ca_array, arterial_delay_meas))
    #plt.plot(t_array, C_array, marker='.', markersize=3, linestyle='', label='measured')
    #plt.plot(t_array, Kety([Ktrans_ref, ve_ref], t_array, ca_array, arterial_delay_ref))
    #plt.show()
    print(['ve meas vs ref ' + str(ve_meas) + ' vs ' + str(ve_ref)])
    print(['Kt meas vs ref ' + str(Ktrans_meas) + ' vs ' + str(Ktrans_ref)])
    #print(['T meas vs ref '+ str(arterial_delay_meas )+ ' vs ' + str(arterial_delay_ref)])

    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve, atol=a_tol_ve)
    np.testing.assert_allclose([Ktrans_meas], [Ktrans_ref], rtol=r_tol_Ktrans, atol=a_tol_Ktrans)
    #np.testing.assert_allclose([arterial_delay_meas], [arterial_delay_ref], rtol=r_tol_delay, atol=a_tol_delay)

