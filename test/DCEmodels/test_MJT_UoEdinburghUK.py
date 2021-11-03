import pytest
import numpy as np

from ..helpers import osipi_parametrize
from . import dce_data
import matplotlib.pyplot as plt
from src.original.MJT_UoEdinburghUK import dce_fit, pk_models, aifs
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
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
def test_MJT_UoEdinburghUK_tofts_model(label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref,a_tol_ve, r_tol_ve, a_tol_vp,r_tol_vp,a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay):
    # NOTES:

    # prepare input data - create aif object
    t_array = t_array #/60  - in seconds
    aif = aifs.patient_specific(t_array, ca_array)
    
    # Create model object and initialise parameters
    pk_model = pk_models.extended_tofts(t_array, aif)
    pk_pars_0 = [{'vp': 0.6, 'ps': 0.02, 've': 0.2}]
    weights = np.concatenate([np.zeros(5), np.ones(len(t_array)-5)])
    
    pk_pars, C_t_fit = dce_fit.conc_to_pkp(C_array, pk_model, pk_pars_0, weights)
    
    Ktrans_meas = pk_pars['ps']
    ve_meas = pk_pars['ve']
    vp_meas = pk_pars['vp']
    
    print(['ve meas vs ref '+ str(ve_meas)+' vs '+str(ve_ref)])
    print(['vp meas vs ref '+ str(vp_meas) + ' vs ' +str(vp_ref)])
    print(['Kt meas vs ref '+ str(Ktrans_meas) + ' vs ' + str(Ktrans_ref)])

    data = [[inspect.stack()[0][3],label,ve_ref,vp_ref,Ktrans_ref,ve_meas,vp_meas,Ktrans_meas]]
    columns = ['testname','label','ve_ref','vp_ref','ktrans_ref','ve_meas','vp_meas','ktrans_meas']
    append_to_excel(data, columns)
    
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve, atol=a_tol_ve)
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp, atol=a_tol_vp)
    np.testing.assert_allclose([Ktrans_meas], [Ktrans_ref], rtol=r_tol_Ktrans, atol=a_tol_Ktrans)
