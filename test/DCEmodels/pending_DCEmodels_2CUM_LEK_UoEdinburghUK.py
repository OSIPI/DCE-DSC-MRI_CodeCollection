import numpy as np
from scipy.optimize import curve_fit

from ..helpers import osipi_parametrize
from . import DCEmodels_data
from src.original.LEK_UoEdinburghUK.PharmacokineticModelling.models import TwoCUM

arg_names = 'label, t_array, C_t_array, cp_aif_array, vp_ref, fp_ref,'\
    'ps_ref, a_tol_vp, r_tol_vp, a_tol_fp, r_tol_fp,'\
    'a_tol_ps, r_tol_ps'
test_data = (DCEmodels_data.dce_DRO_data_2cum())

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_LEK_UoEdinburghUK_2cum_model(label, t_array, C_t_array,
                                      cp_aif_array, vp_ref, fp_ref,
                                      ps_ref, a_tol_vp, r_tol_vp, a_tol_fp,
                                      r_tol_fp, a_tol_ps, r_tol_ps):
    # NOTES:
    # Source code assumes first time point is 0.
    # For testing, the time array is shifted to make this so.
    # Fitting not implemented

    # prepare input data - create aif object
    t_array -= t_array[0]
    t_array /= 60  # convert to mins, so that ps is in /min
    X0 = (0.01, 20/100, 0.15)  # vp, fp, E starting values
    bounds = ((0, 0, 0), (1, 200/100, 1))

    # run test
    output, pcov = curve_fit(lambda t, vp, fp, E: TwoCUM([E, fp, vp],
                                                             t_array, cp_aif_array,
                                                             toff=0),
                             t_array, C_t_array, p0=X0, bounds=bounds)
    vp_meas, fp_meas, E_meas = output
    ps_meas = E_meas*fp_meas / (1-E_meas)
    fp_meas *= 100  # convert from ml/ml/min to ml/100ml/min
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp,
                               atol=a_tol_vp)
    np.testing.assert_allclose([fp_meas], [fp_ref], rtol=r_tol_fp,
                               atol=a_tol_fp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps,
                               atol=a_tol_ps)