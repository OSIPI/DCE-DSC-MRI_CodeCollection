import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from ..helpers import osipi_parametrize
from . import DCEmodels_data
from src.original.LEK_UoEdinburghUK.PharmacokineticModelling.models import \
    TwoCXM

arg_names = 'label, t_array, C_t_array, cp_aif_array, vp_ref, ve_ref, fp_ref,' \
            'ps_ref, delay_ref, a_tol_vp, r_tol_vp, a_tol_ve, r_tol_ve, ' \
            'a_tol_fp, r_tol_fp, a_tol_ps, r_tol_ps, a_tol_delay, r_tol_delay'

test_data = (DCEmodels_data.dce_DRO_data_2cxm())
# Use the test data to generate a parametrize decorator. This causes the
# following test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_LEK_UoEdinburghUK_2cxm_model(label, t_array, C_t_array,
                                      cp_aif_array, vp_ref, ve_ref,
                                      fp_ref, ps_ref, delay_ref,
                                      a_tol_vp, r_tol_vp, a_tol_ve,
                                      r_tol_ve, a_tol_fp, r_tol_fp,
                                      a_tol_ps, r_tol_ps, a_tol_delay,
                                      r_tol_delay):
    # NOTES:
    # Source code assumes first time point is 0.
    # For testing, the time array is shifted to make this so.
    # Simple discrete convolution is used and first term (tau=0) is not scaled.
    # For tests to pass, concentrations are interpolated to temporal
    # resolution of 0.1s
    # Fitting not implemented.

    # prepare input data
    t_array -= t_array[0]  # make first time point = 0
    t_array /= 60  # convert to mins, so that Fp and PS are in /min
    t_interp = np.arange(0, 299.5, 0.1) / 60  # interpolate
    c_ap_func = interp1d(t_array, cp_aif_array, kind='quadratic',
                         bounds_error=False, fill_value=(0, cp_aif_array[-1]))
    C_t_func = interp1d(t_array, C_t_array, kind='quadratic',
                        bounds_error=False, fill_value=(0, C_t_array[-1]))
    cp_aif_interp = c_ap_func(t_interp)
    C_t_interp = C_t_func(t_interp)
    X0 = (0.01, 0.2, 20 / 100, 0.15)  # vp, ve, Fp, E starting values
    bounds = ((0, 0, 0, 0), (1, 1, 200 / 100, 1))

    # run test
    output, pcov = curve_fit(lambda t, vp, ve, fp, E: TwoCXM([E, fp, ve, vp],
                                                             t, cp_aif_interp,
                                                             toff=0),
                             t_interp, C_t_interp, p0=X0, bounds=bounds)
    vp_meas, ve_meas, fp_meas, E_meas = output
    ps_meas = E_meas * fp_meas / (1 - E_meas)
    fp_meas *= 100  # convert from ml/ml/min to ml/100ml/min

    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp,
                               atol=a_tol_vp)
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve,
                               atol=a_tol_ve)
    np.testing.assert_allclose([fp_meas], [fp_ref], rtol=r_tol_fp,
                               atol=a_tol_fp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps,
                               atol=a_tol_ps)


test_data_delay = (DCEmodels_data.dce_DRO_data_2cxm(delay=True))
@osipi_parametrize(arg_names, test_data_delay, xf_labels=[])
def test_LEK_UoEdinburghUK_2cxm_model_delay(label, t_array, C_t_array,
                                            cp_aif_array, vp_ref, ve_ref,
                                            fp_ref, ps_ref, delay_ref,
                                            a_tol_vp, r_tol_vp, a_tol_ve,
                                            r_tol_ve, a_tol_fp, r_tol_fp,
                                            a_tol_ps, r_tol_ps, a_tol_delay,
                                            r_tol_delay):
    # NOTES:
    # Source code assumes first time point is 0.
    # For testing, the time array is shifted to make this so.
    # Simple discrete convolution is used and first term (tau=0) is not scaled.
    # For tests to pass, concentrations are interpolated to temporal
    # resolution of 0.1s
    # Fitting not implemented.

    # prepare input data
    t_array -= t_array[0]  # make first time point = 0
    t_array /= 60  # convert to mins, so that Fp and PS are in /min
    t_interp = np.arange(0, 299.5, 0.1) / 60  # interpolate
    c_ap_func = interp1d(t_array, cp_aif_array, kind='quadratic',
                         bounds_error=False, fill_value=(0, cp_aif_array[-1]))
    C_t_func = interp1d(t_array, C_t_array, kind='quadratic',
                        bounds_error=False, fill_value=(0, C_t_array[-1]))
    cp_aif_interp = c_ap_func(t_interp)
    C_t_interp = C_t_func(t_interp)
    X0 = (0.01, 0.2, 20/100, 0.15, 0)  # vp, ve, Fp, E, delay starting values
    bounds = ((0, 0, 0, 0, -10/60), (1, 1, 200/100, 1, 10/60))

    # run test
    output, pcov = curve_fit(lambda t, vp, ve, fp, E, delay: TwoCXM([E, fp, ve,
                                                                     vp],
                                                                    t,
                                                                    cp_aif_interp,
                                                                    toff=delay),
                             t_interp, C_t_interp, p0=X0, bounds=bounds)
    vp_meas, ve_meas, fp_meas, E_meas, delay_meas = output
    ps_meas = E_meas * fp_meas / (1 - E_meas)
    fp_meas *= 100  # convert from ml/ml/min to ml/100ml/min
    delay_meas *= 60  # convert to s

    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp,
                               atol=a_tol_vp)
    np.testing.assert_allclose([ve_meas], [ve_ref], rtol=r_tol_ve,
                               atol=a_tol_ve)
    np.testing.assert_allclose([fp_meas], [fp_ref], rtol=r_tol_fp,
                               atol=a_tol_fp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps,
                               atol=a_tol_ps)
    np.testing.assert_allclose([delay_meas], [delay_ref], rtol=r_tol_delay,
                               atol=a_tol_delay)
