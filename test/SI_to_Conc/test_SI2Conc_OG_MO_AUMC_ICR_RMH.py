import pytest
import numpy as np

from ..helpers import osipi_parametrize
from . import SI2Conc_data
from src.original.OG_MO_AUMC_ICR_RMH.ExtendedTofts.DCE import dce_to_r1eff
from src.original.OG_MO_AUMC_ICR_RMH.ExtendedTofts.DCE import r1eff_to_conc



# All tests will use the same arguments and same data...
arg_names = 'label', 'fa', 'tr', 'T1base', 'BLpts', 'r1', 's_array', 'conc_array', 'a_tol', 'r_tol'
test_data = SI2Conc_data.SI2Conc_data()


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels = [])
def test_OG_MO_AUMC_ICR_RMH_dce_to_r1eff(label, fa, tr, T1base, BLpts, r1, s_array, conc_array, a_tol, r_tol):

    ##Prepare input data
    #Convert fa to radians
    fa_rad=fa * np.pi/180.
    #This function uses a value for S0, which would be the mean of the s_array from 1 to BLpts.
    # It's from 1 rather than 0 because the code used to do the original conversion skips the first point of the SI curve
    #s0=np.mean(s_array[1:BLpts])

    #This function expects a signal array of shape (x,1) rather than (x,) s0 add another dimension to the signal array to make it (150,1) rather than (150,) 
    s_array=s_array[:,None].T
    
    # run test
    #The code uses two functions to get from SI to conc
    r1_curve = dce_to_r1eff(s_array, 1/T1base, tr, fa_rad, BLpts)
    conc_curve = r1eff_to_conc(r1_curve, 1/T1base, r1)


    np.testing.assert_allclose( conc_curve, [conc_array], rtol=r_tol, atol=a_tol )

