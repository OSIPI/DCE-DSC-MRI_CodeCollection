import os
import numpy as np
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
from . import DCEmodels_data
from src.original.MJT_UoEdinburgh_UK import dce_fit, pk_models, aifs

arg_names = (
    "label, t_array, C_t_array, cp_aif_array, vp_ref, ps_ref, "
    "delay_ref, a_tol_vp, r_tol_vp, a_tol_ps, r_tol_ps, a_tol_delay, "
    "r_tol_delay"
)
test_data = DCEmodels_data.dce_DRO_data_Patlak()

filename_prefix = ""


def setup_module(module):
    # initialize the logfiles
    global filename_prefix  # we want to change the global variable
    os.makedirs("./test/results/DCEmodels", exist_ok=True)
    filename_prefix = "DCEmodels/TestResults_models"
    log_init(
        filename_prefix,
        "_MJT_UoEdinburgh_UK_patlak",
        [
            "label",
            "time (us)",
            "vp_ref",
            "ps_ref",
            "delay_ref",
            "vp_meas",
            "ps_meas",
            "delay_meas",
        ],
    )
    log_init(
        filename_prefix,
        "_MJT_UoEdinburgh_UK_patlak_llsq",
        [
            "label",
            "time (us)",
            "vp_ref",
            "ps_ref",
            "delay_ref",
            "vp_meas",
            "ps_meas",
            "delay_meas",
        ],
    )


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_MJT_UoEdinburgh_UK_Patlak_model(
    label,
    t_array,
    C_t_array,
    cp_aif_array,
    vp_ref,
    ps_ref,
    delay_ref,
    a_tol_vp,
    r_tol_vp,
    a_tol_ps,
    r_tol_ps,
    a_tol_delay,
    r_tol_delay,
):
    # NOTES:

    # prepare input data - create aif object
    aif = aifs.PatientSpecific(t_array, cp_aif_array)
    pk_model = pk_models.Patlak(t_array, aif, upsample_factor=3)

    # run code
    tic = perf_counter()
    vp_meas, ps_meas, C_t_fit = dce_fit.ConcToPKP(pk_model).proc(C_t_array)
    exc_time = 1e6 * (perf_counter() - tic)  # measure execution time

    # log results
    log_results(
        filename_prefix,
        "_MJT_UoEdinburgh_UK_patlak",
        [
            [
                label,
                f"{exc_time:.0f}",
                vp_ref,
                ps_ref,
                delay_ref,
                vp_meas,
                ps_meas,
                delay_ref,
            ]
        ],
    )

    # run test
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp, atol=a_tol_vp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps, atol=a_tol_ps)


test_data_delay = DCEmodels_data.dce_DRO_data_Patlak(delay=True)


# Use the test data to generate a parametrize decorator. This causes the
# following test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data_delay, xf_labels=[])
def test_MJT_UoEdinburgh_UK_Patlak_model_delay(
    label,
    t_array,
    C_t_array,
    cp_aif_array,
    vp_ref,
    ps_ref,
    delay_ref,
    a_tol_vp,
    r_tol_vp,
    a_tol_ps,
    r_tol_ps,
    a_tol_delay,
    r_tol_delay,
):
    # NOTES:

    # prepare input data - create aif object
    aif = aifs.PatientSpecific(t_array, cp_aif_array)
    pk_model = pk_models.Patlak(t_array, aif, upsample_factor=3, fixed_delay=None)

    # run code
    tic = perf_counter()
    vp_meas, ps_meas, delay_meas, C_t_fit = dce_fit.ConcToPKP(pk_model).proc(C_t_array)
    exc_time = 1e6 * (perf_counter() - tic)  # measure execution time

    # log results
    log_results(
        filename_prefix,
        "_MJT_UoEdinburgh_UK_patlak",
        [
            [
                label,
                f"{exc_time:.0f}",
                vp_ref,
                ps_ref,
                delay_ref,
                vp_meas,
                ps_meas,
                delay_meas,
            ]
        ],
    )

    # run test
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp, atol=a_tol_vp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps, atol=a_tol_ps)
    np.testing.assert_allclose(
        [delay_meas], [delay_ref], rtol=r_tol_delay, atol=a_tol_delay
    )


@osipi_parametrize(arg_names, test_data, xf_labels=[])
def test_MJT_UoEdinburgh_UK_Patlak_model_llsq(
    label,
    t_array,
    C_t_array,
    cp_aif_array,
    vp_ref,
    ps_ref,
    delay_ref,
    a_tol_vp,
    r_tol_vp,
    a_tol_ps,
    r_tol_ps,
    a_tol_delay,
    r_tol_delay,
):
    # NOTES:

    # prepare input data - create aif object
    aif = aifs.PatientSpecific(t_array, cp_aif_array)

    # run code
    tic = perf_counter()
    vp_meas, ps_meas, C_t_fit = dce_fit.PatlakLinear(
        t_array, aif, upsample_factor=3
    ).proc(C_t_array)
    exc_time = 1e6 * (perf_counter() - tic)  # measure execution time

    # log results
    log_results(
        filename_prefix,
        "_MJT_UoEdinburgh_UK_patlak_llsq",
        [
            [
                label,
                f"{exc_time:.0f}",
                vp_ref,
                ps_ref,
                delay_ref,
                vp_meas,
                ps_meas,
                delay_ref,
            ]
        ],
    )

    # run test
    np.testing.assert_allclose([vp_meas], [vp_ref], rtol=r_tol_vp, atol=a_tol_vp)
    np.testing.assert_allclose([ps_meas], [ps_ref], rtol=r_tol_ps, atol=a_tol_ps)
