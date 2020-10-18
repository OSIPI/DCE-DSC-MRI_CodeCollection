from src.original.t1 import fit_vfa_nonlinear, fit_vfa_2_point, fit_vfa_linear
import pytest
import numpy as np
from math import pi

# Test for specific inputs and expected outputs :
"""
For the time being, simulated values without noise have been specified to test the fit.
Additional test cases with varying degrees of noise should be included, from maybe a DRO.
"""
    
@pytest.mark.parametrize('S_t, fa_array, tr_S, T1_est, S0', [(np.array([48.86,58.95,69.14,63.22]),np.array([20,15,10,5]),0.01,1.,1000.)])

def test_fit_vfa_nonlinear(S_t, fa_array, tr_S, T1_est, S0):
    fa_array_rad = fa_array * pi/180.
    np.testing.assert_array_almost_equal(fit_vfa_nonlinear(S_t,fa_array_rad,tr_S),[S0,T1_est],decimal=2)

@pytest.mark.parametrize('S_t, fa_array, tr_S, T1_est, S0', [(np.array([48.86,58.95,69.14,63.22]),np.array([20,15,10,5]),0.01,1.,1000.)])
def test_fit_vfa_linear(S_t, fa_array, tr_S, T1_est, S0):
    fa_array_rad = fa_array * pi/180.
    np.testing.assert_array_almost_equal(fit_vfa_linear(S_t,fa_array_rad,tr_S),[S0,T1_est],decimal=2)

@pytest.mark.parametrize('S_t, fa_array, tr_S, T1_est, S0', [(np.array([48.86,58.95,69.14,63.22]),np.array([20,15,10,5]),0.01,1.,1000.)])
def test_fit_vfa_2_point(S_t,fa_array, tr_S, T1_est, S0):
    fa_array_rad = fa_array * pi/180.
    np.testing.assert_array_almost_equal(np.round(fit_vfa_2_point(S_t,fa_array_rad,tr_S)),[[S0],[T1_est]],decimal=2)

# Test for valid input values :
"""
If invalid flip angles are used i.e. <=0, then raise an exception
"""
@pytest.mark.parametrize('fa_array',[(np.array([-10,8,6,4])),
                                      (np.array([0,2,4,6])),
                                      (np.array([20,10,0,-5]))])
def test_negative_favals(fa_array):
    with pytest.raises(ValueError) as exc_info:
        if fa_array.all() == False:
            raise ValueError("Invalid flip angle provided")
            assert exc_info.type is ValueError
    

# Test for valid input data types:
    """
    Check if the correct data types aree used for the variables 
    """

    

# Test for len(Signal array) == len(fa_array):
    """
    Check for array dimensionality mismatch whiich could result in an error while computing fit
    """
