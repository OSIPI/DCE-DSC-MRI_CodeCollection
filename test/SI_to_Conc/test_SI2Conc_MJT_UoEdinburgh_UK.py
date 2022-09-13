import os
import numpy as np
from time import perf_counter
from ..helpers import osipi_parametrize, log_init, log_results
from . import SI2Conc_data
from osipi_code_collection.original.MJT_UoEdinburgh_UK.signal_models import SPGR
from osipi_code_collection.original.MJT_UoEdinburgh_UK.relaxivity import CRLinear
from osipi_code_collection.original.MJT_UoEdinburgh_UK.dce_fit import SigToEnh
from osipi_code_collection.original.MJT_UoEdinburgh_UK.dce_fit import EnhToConc, EnhToConcSPGR


# All tests will use the same arguments and same data...
arg_names = 'label', 'fa', 'tr', 'T1base', 'BLpts', 'r1', 's_array', 'conc_array', 'a_tol', 'r_tol'
test_data = SI2Conc_data.SI2Conc_data()

filename_prefix = ''

def setup_module(module):
    # initialize the logfiles
    global filename_prefix # we want to change the global variable
    os.makedirs('./test/results/SI_to_Conc', exist_ok=True)
    filename_prefix = 'SI_to_Conc/TestResults_SI2Conc'
    log_init(filename_prefix, '_MJT_UoEdinburgh', ['label', 'time (us)', 'conc_ref', 'conc_meas'])
    log_init(filename_prefix, '_MJT_UoEdinburgh_num', ['label', 'time (us)', 'conc_ref', 'conc_meas'])

# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels = [])
def test_MJT_UoEdinburgh_UK_sig_to_conc_num(label, fa, tr, T1base, BLpts, r1, s_array, conc_array, a_tol, r_tol):
    # Note: the first signal value is not used for baseline estimation,
    # and the first C value is not logged or assessed

    # Tests code that converts Enh->Conc using a numerical approach
    # This method can be used for different pulse sequences and C->R1 relationships
    # Can account for R2* relaxivity

    # Note: the first signal value is not used for baseline estimation,
    # and the first C value is not logged or assessed

    ##Prepare input data

    #We need an array of integers to tell the function which points are for the baseline.  This will be 1:BLpts
    BL_idx=np.arange(1,BLpts,1)

    #We need to set the signal model to be spgr which will require the tr, the flip angle and a dummy te (ignored by default)
    sigmod=SPGR(tr, fa, 1)
    #We also need to set the c_to_r_model to be linear, with relaxivity r1 and dummy r2 relaxivity
    conc_to_r=CRLinear(r1,0)


    # run test
    #The code uses two functions to get from SI to conc
    tic = perf_counter()
    enh_curve = SigToEnh(BL_idx).proc(s_array)

    #B1 correction factor k=1
    conc_curve = EnhToConc(conc_to_r, sigmod).proc(enh_curve, T1base)
    exc_time = 1e6 * (perf_counter() - tic)

    # log results
    row_data = []
    for ref, meas in zip(conc_array[1:], conc_curve[1:]):
        row_data.append([label, f"{exc_time:.0f}", ref, meas])
    log_results(filename_prefix, '_MJT_UoEdinburgh_num', row_data)

    # testing
    np.testing.assert_allclose( conc_curve[1:], conc_array[1:], rtol=r_tol, atol=a_tol)


@osipi_parametrize(arg_names, test_data, xf_labels = [])
def test_MJT_UoEdinburgh_UK_sig_to_conc(label, fa, tr, T1base, BLpts, r1, s_array, conc_array, a_tol, r_tol):
    # Note: the first signal value is not used for baseline estimation,
    # and the first C value is not logged or assessed

    # Tests code that calculates Enh->Conc analytically for the SPGR sequence
    # Requires linear relaxivity relationship and assumes no R2* relaxation

    ##Prepare input data

    #We need an array of integers to tell the function which points are for the baseline.  This will be 1:BLpts
    BL_idx=np.arange(1,BLpts,1)

    # run test
    #The code uses two functions to get from SI to conc
    tic = perf_counter()
    enh_curve = SigToEnh(BL_idx).proc(s_array)
    #B1 correction factor k=1
    conc_curve = EnhToConcSPGR(tr, fa, r1).proc(enh_curve, T1base)
    exc_time = 1e6 * (perf_counter() - tic)

    # log results
    row_data = []
    for ref, meas in zip(conc_array[1:], conc_curve[1:]):
        row_data.append([label, f"{exc_time:.0f}", ref, meas])
    log_results(filename_prefix, '_MJT_UoEdinburgh', row_data)

    # testing
    np.testing.assert_allclose( conc_curve[1:], conc_array[1:], rtol=r_tol, atol=a_tol)