import pytest
import numpy as np

from ..helpers import osipi_parametrize
from . import dce_data
from src.original.LCB_BNI.dce import fit_tofts

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
    Ktrans_meas, ve_meas, Ct_fit = fit_tofts(t_array, C_array, ca_array)

    print(['ve meas vs ref ' + str(ve_meas) + ' vs ' + str(ve_ref)])
    print(['Kt meas vs ref ' + str(Ktrans_meas) + ' vs ' + str(Ktrans_ref)])

    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve, atol=a_tol_ve)
    np.testing.assert_allclose([Ktrans_meas], [Ktrans_ref], rtol=r_tol_Ktrans, atol=a_tol_Ktrans)
