import pytest
import numpy as np

from ..helpers import osipi_parametrize
from . import dce_data
import matplotlib.pyplot as plt
from src.original.MJT_UoEdinburghUK import dce_fit, pk_models, aifs
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import inspect

# All tests will use the same arguments and same data...
arg_names = 'label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref,  a_tol_ve, r_tol_ve, a_tol_vp,r_tol_vp,a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay'
test_data = (
    dce_data.dce_DRO_data()
    )


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
# In the following test, we specify 1 case that is expected to fail...
@osipi_parametrize(arg_names, test_data, xf_labels = [])
def test_MJT_UoEdinburghUK_tofts_model(label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref,a_tol_ve, r_tol_ve, a_tol_vp,r_tol_vp,a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay):
    # NOTES:

    # prepare input data - create aif object
    t_array = t_array #/60  - in seconds
    aif = aifs.patient_specific(t_array, ca_array)
    
    # Create model object and initialise parameters
    pk_model = pk_models.extended_tofts(t_array, aif)
    pk_pars_0 = [{'vp': 0.6, 'ps': 0.02, 've': 0.2}]
    #weights = np.concatenate([np.zeros(5), np.ones(len(t_array)-5)])
    
    pk_pars, C_t_fit = dce_fit.conc_to_pkp(C_array, pk_model, pk_pars_0)
    
    Ktrans_meas = pk_pars['ps']
    ve_meas = pk_pars['ve']
    vp_meas = pk_pars['vp']
    
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

    # prepare input data - create aif object
    t_array = t_array  # /60  - in seconds
    aif = aifs.patient_specific(t_array, ca_array)

    # Create model object and initialise parameters
    pk_model = pk_models.tofts(t_array, aif)
    #pk_pars_0 = [{'ktrans': 0.02, 've': 0.2}]
    # weights = np.concatenate([np.zeros(5), np.ones(len(t_array)-5)])

    pk_pars, C_t_fit = dce_fit.conc_to_pkp(C_array, pk_model)

    Ktrans_meas = pk_pars['ktrans']
    ve_meas = pk_pars['ve']

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