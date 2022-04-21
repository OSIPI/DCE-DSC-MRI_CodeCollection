import numpy as np

from ..helpers import osipi_parametrize
from . import DCEmodels_data
from src.original.OG_MO_AUMC_ICR_RMH.ExtendedTofts.DCE import fit_tofts_model, fit_aif

# All tests will use the same arguments and same data...
arg_names = 'label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref,  a_tol_ve, ' \
            'r_tol_ve, a_tol_vp,r_tol_vp,a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay '

test_data = (DCEmodels_data.dce_DRO_data_extended_tofts_kety())
# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def testOG_MO_AUMC_ICR_RMH_extended_tofts_kety_model(label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref,
                                                     Ktrans_ref, arterial_delay_ref, a_tol_ve, r_tol_ve, a_tol_vp,
                                                     r_tol_vp, a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay):
    # NOTES:
    # Not possible to fit without a delay.
    # Therefore, delay constraint range was set to zero.

    # prepare input data
    ta_array = ta_array / 60
    t_array = t_array / 60
    arterial_delay_ref = arterial_delay_ref / 60
    x0 = (0.6/0.2, 0, 0.2, 0.01)  # ke, delay, ve, vp
    bounds = ((0.0, -0, 0.0, 0.0), (5.0/0.2, 0, 1, 1))
    try:
        AIF = fit_aif(ca_array, ta_array, model='Cosine8')
    except:
        AIF = fit_aif(ca_array, ta_array, model='Cosine4')
    C_array = np.array(C_array)
    C_array = C_array[np.newaxis, ...]

    # run test
    try:
        ke, arterial_delay_meas, ve_meas, vp_meas = fit_tofts_model(C_array, t_array, AIF, idxs=None,
                                                                    X0=(0.6, 0.2, 0.2, 0.02),
                                                                    bounds=((0.0, 0, 0.0, 0.0), (5.0, 1, 0.7, 0.7)),
                                                                    jobs=1, model='Cosine8')
    except:
        ke, arterial_delay_meas, ve_meas, vp_meas = fit_tofts_model(C_array, t_array, AIF, idxs=None,
                                                                    X0=(0.6, 0.2, 0.2, 0.02),
                                                                    bounds=((0.0, 0, 0.0, 0.0), (5.0, 1, 0.7, 0.7)),
                                                                    jobs=1, model='Cosine4')
    Ktrans_meas = ke * ve_meas
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve, atol=a_tol_ve)
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp, atol=a_tol_vp)
    np.testing.assert_allclose([Ktrans_meas], [Ktrans_ref], rtol=r_tol_Ktrans, atol=a_tol_Ktrans)


test_data_delay = (DCEmodels_data.dce_DRO_data_extended_tofts_kety(delay=True))
# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data_delay, xf_labels=[])
def testOG_MO_AUMC_ICR_RMH_extended_tofts_kety_model_delay(label, t_array,
                                                      C_array, ca_array, ta_array, ve_ref, vp_ref,
                                                     Ktrans_ref, arterial_delay_ref, a_tol_ve, r_tol_ve, a_tol_vp,
                                                     r_tol_vp, a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay):
    # NOTES:

    # prepare input data
    ta_array = ta_array / 60
    t_array = t_array / 60
    arterial_delay_ref = arterial_delay_ref / 60
    x0 = (0.6/0.2, 0, 0.2, 0.01)  # ke, delay, ve, vp
    bounds = ((0.0, -10/60, 0.0, 0.0), (5.0/0.2, 10/60, 1, 1))
    try:
        AIF = fit_aif(ca_array, ta_array, model='Cosine8')
    except:
        AIF = fit_aif(ca_array, ta_array, model='Cosine4')
    C_array = np.array(C_array)
    C_array = C_array[np.newaxis, ...]

    # run test
    try:
        ke, arterial_delay_meas, ve_meas, vp_meas = fit_tofts_model(C_array, t_array, AIF, idxs=None,
                                                                    X0=(0.6, 0.2, 0.2, 0.02),
                                                                    bounds=((0.0, 0, 0.0, 0.0), (5.0, 1, 0.7, 0.7)),
                                                                    jobs=1, model='Cosine8')
    except:
        ke, arterial_delay_meas, ve_meas, vp_meas = fit_tofts_model(C_array, t_array, AIF, idxs=None,
                                                                    X0=(0.6, 0.2, 0.2, 0.02),
                                                                    bounds=((0.0, 0, 0.0, 0.0), (5.0, 1, 0.7, 0.7)),
                                                                    jobs=1, model='Cosine4')
    Ktrans_meas = ke * ve_meas
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve, atol=a_tol_ve)
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp, atol=a_tol_vp)
    np.testing.assert_allclose([Ktrans_meas], [Ktrans_ref], rtol=r_tol_Ktrans, atol=a_tol_Ktrans)
    np.testing.assert_allclose([arterial_delay_meas], [arterial_delay_ref], rtol=r_tol_delay, atol=a_tol_delay)
