import pytest
import numpy as np
from math import pi
from lmfit import Model
import pandas as pd

import t1_data
from src.original.ST_SydneyAus.VFAT1mapping import spgr_linear, spgr_nonlinear, VFAT1mapping


# Test for specific inputs and expected outputs :
"""

"""

@t1_data.parameters
def test_ST_SydneyAus(label, fa_array, tr_array, s_array, r1_ref, s0_ref):
    # prepare input data
    tr = tr_array[0] * 1000. # convert s to ms
    t1_ref = (1./r1_ref) * 1000. # convert R1 in /s to T1 in ms
 
    # run tests
    [s0_meas, t1_meas] = VFAT1mapping( fa_array, s_array, tr, method = 'nonlinear' )
    r1_meas = 1000./t1_meas
    
    np.testing.assert_allclose( [s0_meas, r1_meas], [s0_ref, r1_ref], rtol=0.01, atol=0 )



# Test another T1 mapping contribution, as above


# Test for valid input data types, array dimensionality etc:
    """
    TO BE IMPLEMENTED IN A FUTURE MILESTONE
    """