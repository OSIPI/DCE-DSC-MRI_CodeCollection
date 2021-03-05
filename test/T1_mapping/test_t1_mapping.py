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


# Test src.original.ST_SydneyAus.VFAT1mapping.VFAT1mapping
@pytest.mark.parametrize('fa_array, tr_array, s_array, r1_ref, s0_ref',
                         t1_data.t1_brain_data() +
                         t1_data.t1_prostate_data() +
                         t1_data.t1_some_more_data()
                         )
def test_ST_SydneyAus(fa_array, tr_array, s_array, r1_ref, s0_ref):
    # prepare data
    tr = tr_array[0] * 1000.
    t1_ref = (1./r1_ref) * 1000.
    
    np.testing.assert_allclose( #non-linear
        VFAT1mapping( fa_array, s_array, tr, method = 'nonlinear' ),
        [s0_ref,t1_ref], rtol=0.01, atol=0 )

    np.testing.assert_allclose( #linear
        VFAT1mapping( fa_array, s_array, tr, method = 'linear' ),
        [s0_ref,t1_ref], rtol=0.05, atol=0 )


# Test another T1 mapping contribution
@pytest.mark.parametrize('fa_array, tr_array, s_array, r1_ref, s0_ref',
                         t1_data.t1_brain_data() +
                         t1_data.t1_prostate_data() +
                         t1_data.t1_some_more_data()
                         )
def test_another_contribution(fa_array, tr_array, s_array, r1_ref, s0_ref):
    # as above...
    pass


# Test for valid input data types, array dimensionality etc:
    """
    TO BE IMPLEMENTED IN A FUTURE MILESTONE
    """