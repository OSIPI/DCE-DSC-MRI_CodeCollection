import pytest
import numpy as np

from ..helpers import osipi_parametrize
from . import dce_data
import matplotlib.pyplot as plt
from src.original.OG_MO_AUMC_ICR_RMH.ExtendedTofts.DCE import Cosine4AIF, Cosine4AIF_ExtKety, Cosine8AIF, Cosine8AIF_ExtKety, fit_tofts_model, fit_aif
import pandas as pd
import inspect
from .tools import append_to_excel

# All tests will use the same arguments and same data...
arg_names = 'label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref,  a_tol_ve, r_tol_ve, a_tol_vp,r_tol_vp,a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay'
# test_data = (
#     dce_data.dce_DRO_data() +
#     dce_data.dce_DRO_data(delay=True)
#     )
test_data = (
    dce_data.dce_DRO_data()
    )

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
# In the following test, we specify 1 case that is expected to fail...
@osipi_parametrize(arg_names, test_data, xf_labels = ['test_vox_WM', 'test_vox_GM','test_vox_WM_10','test_vox_GM_10','test_vox_WM_20','test_vox_GM_20','test_vox_WM_30','test_vox_GM_30','test_vox_WM_50','test_vox_GM_50'])
def testOG_MO_AUMC_ICR_RMH_tofts_model(label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref, a_tol_ve, r_tol_ve, a_tol_vp,r_tol_vp,a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay):
    # NOTES:

    # prepare input data
    ta_array = ta_array/60
    t_array = t_array/60
    arterial_delay_ref = arterial_delay_ref/60
    try:
        AIF=fit_aif(ca_array, ta_array, model='Cosine8')
        #AIF['t0']=0
        #plt.plot(ta_array,
        #         Cosine8AIF(ta_array, AIF['ab'], AIF['ar'], AIF['ae'], AIF['mb'], AIF['mm'], AIF['mr'], AIF['me'],
        #                        AIF['tr'], AIF['t0']),label='Model fit to the measured AIF in the abdomen (MRM/Mol Onc.)')  # https://doi.org/10.1016/j.mri.2018.02.005 and used in https://doi.org/10.1002/1878-0261.12688
        #plt.plot(ta_array, ca_array, marker='.', markersize=3, linestyle='', label='measured')
    except:
        AIF = fit_aif(ca_array, ta_array, model='Cosine4')
        # AIF['t0']=0
        #plt.plot(ta_array, Cosine4AIF(ta_array, AIF['ab'], AIF['ae'], AIF['mb'], AIF['me'], AIF['t0']),
        #         label='Population AIF from H&N study JNM')  # https://doi.org/10.2967/jnumed.116.174433
        #plt.plot(ta_array, ca_array, marker='.', markersize=3, linestyle='', label='measured')
    #plt.legend()
    #plt.show()
    C_array=np.array(C_array)
    C_array=C_array[np.newaxis,...]
    # run test
    try:
        ke, arterial_delay_meas, ve_meas, vp_meas = fit_tofts_model(C_array, t_array, AIF, idxs=None, X0=(0.6, 0.2, 0.2, 0.02), bounds=((0.0, 0, 0.0, 0.0), (5.0, 1, 0.7, 0.7)),
                        jobs=4, model='Cosine8')
    except:
        ke, arterial_delay_meas, ve_meas, vp_meas = fit_tofts_model(C_array, t_array, AIF, idxs=None, X0=(0.6, 0.2, 0.2, 0.02), bounds=((0.0, 0, 0.0, 0.0), (5.0, 1, 0.7, 0.7)),
                        jobs=4, model='Cosine4')
    # ke_ref=Ktrans_ref/(ve_ref+0.00001)
    #
    # plt.plot(t_array, np.squeeze(C_array), marker='x', markersize=3, linestyle='',
    #          label='Measured signal')
    # try:
    #     simdat = Cosine8AIF_ExtKety(t_array, AIF, ke, arterial_delay_meas, ve_meas, vp_meas)
    #     plt.plot(t_array, simdat, marker='', markersize=3, linestyle='-',
    #              label='fitted')
    #     simdat = Cosine8AIF_ExtKety(t_array, AIF, ke_ref+0.00000001, arterial_delay_ref, ve_ref, vp_ref)
    #     plt.plot(t_array, simdat, marker='', markersize=3, linestyle='-',
    #              label='ref')
    # except:
    #     simdat = Cosine4AIF_ExtKety(t_array, AIF, ke, arterial_delay_meas, ve_meas, vp_meas)
    #     plt.plot(t_array, simdat, marker='', markersize=3, linestyle='-',
    #              label='fitted')
    #     simdat = Cosine4AIF_ExtKety(t_array, AIF, ke_ref+0.00000001, arterial_delay_ref, ve_ref, vp_ref)
    #     plt.plot(t_array, simdat, marker='', markersize=3, linestyle='-',
    #              label='ref')
    # plt.legend()
    # plt.show()
    Ktrans_meas = ke * ve_meas

    print(['ve meas vs ref '+ str(ve_meas)+' vs '+str(ve_ref)])
    print(['vp meas vs ref '+ str(vp_meas) + ' vs ' +str(vp_ref)])
    print(['Kt meas vs ref '+ str(Ktrans_meas) + ' vs ' + str(Ktrans_ref)])
    print(['T meas vs ref '+ str(arterial_delay_meas )+ ' vs ' + str(arterial_delay_ref)])
    
    data = [[inspect.stack()[0][3],label,ve_ref,vp_ref,Ktrans_ref,ve_meas,vp_meas,Ktrans_meas]]
    columns = ['testname','label','ve_ref','vp_ref','ktrans_ref','ve_meas','vp_meas','ktrans_meas']
    append_to_excel(data, columns)
        
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve, atol=a_tol_ve)
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp, atol=a_tol_vp)
    np.testing.assert_allclose([Ktrans_meas], [Ktrans_ref], rtol=r_tol_Ktrans, atol=a_tol_Ktrans)
    np.testing.assert_allclose([arterial_delay_meas], [arterial_delay_ref], rtol=r_tol_delay, atol=a_tol_delay)
