import numpy as np
from scipy.optimize import curve_fit

from ..helpers import osipi_parametrize
from . import DCEmodels_data
from src.original.LEK_UoEdinburghUK.PharmacokineticModelling.models import Patlak

arg_names = 'label, t_array, C_t_array, cp_aif_array, vp_ref, ps_ref, a_tol_vp,\
            r_tol_vp, a_tol_ps, r_tol_ps'
test_data = (DCEmodels_data.dce_DRO_data_Patlak())

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_LEK_UoEdinburghUK_Patlak_model(label, t_array, C_t_array, cp_aif_array,
                                        vp_ref, ps_ref, a_tol_vp, r_tol_vp,
                                        a_tol_ps, r_tol_ps):
    # NOTES:

    # prepare input data
    t_array = t_array / 60  # convert to mins, so that ps is in /min
    X0 = (0.1, 0.1)  # ps, vp starting values
    bounds = ((0.0, 0.0), (5.0, 1.0))

    # run test
    output, pcov = curve_fit(lambda t, x, y: Patlak([x, y], t, cp_aif_array,
                                                    toff=0),
                             t_array, C_t_array, p0=X0,
                             bounds=bounds)
    ps_meas, vp_meas = output
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp,
                               atol=a_tol_vp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps,
                               atol=a_tol_ps)
