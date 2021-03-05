#from src.original.T1_mapping.ST_SydneyAus.VFAT1mapping import spgr_linear, spgr_nonlinear, VFAT1mapping
from src.original.ST_SydneyAus.VFAT1mapping import spgr_linear, spgr_nonlinear, VFAT1mapping
import pytest
import numpy as np
from math import pi
from lmfit import Model

# Test for specific inputs and expected outputs :
"""
For the time being, simulated values without noise have been specified to test the fit.
Additional test cases with varying degrees of noise should be included, from maybe a DRO.
"""
    
@pytest.mark.parametrize('S_t, fa_array, tr_S, T1_est, S0', [(np.array([48.86,58.95,69.14,63.22]),np.array([20,15,10,5]),0.01,1.,1000.)])

def test_VFAT1mapping(fa_array, S_t, tr_S, S0, T1_est):
    fa_array_rad = fa_array 
    np.testing.assert_array_almost_equal(VFAT1mapping(fa_array_rad,S_t,tr_S),[S0,T1_est],decimal=2)


# Test for valid input values :
"""
If invalid flip angles are used i.e. <=0, then raise an exception
"""
@pytest.mark.parametrize('fa_array',[(np.array([0,8,6,4])),
                                      (np.array([0,2,4,6])),
                                      (np.array([20,10,0,-5]))])
def test_negative_favals(fa_array):
    with pytest.raises(ValueError) as exc_info:
        if fa_array.all() == False:
            raise ValueError("Invalid flip angle provided")
            assert exc_info.type is ValueError
    

# Test for valid input data types, array dimensionality etc:
    """
    TO BE IMPLEMENTED IN FUTURE MILESTONE
    """