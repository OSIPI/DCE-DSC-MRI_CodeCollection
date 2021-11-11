import pytest
import numpy as np

from ..helpers import osipi_parametrize
from . import SI2Conc_data
from src.original.MJT_UoEdinburghUK.signal_models import spgr
from src.original.MJT_UoEdinburghUK.relaxivity import c_to_r_linear
from src.original.MJT_UoEdinburghUK.dce_fit import sig_to_enh
from src.original.MJT_UoEdinburghUK.dce_fit import enh_to_conc



# All tests will use the same arguments and same data...
arg_names = 'label', 'fa', 'tr', 'T1base', 'BLpts', 'r1', 's_array', 'conc_array', 'a_tol', 'r_tol'
test_data = SI2Conc_data.SI2Conc_data()


# Use the test data to generate a parametrize decorator. This causes the following
# test to be run for every test case listed in test_data...
@osipi_parametrize(arg_names, test_data, xf_labels = [])
def test_MJT_UoEdinburghUK_sig_to_conc(label, fa, tr, T1base, BLpts, r1, s_array, conc_array, a_tol, r_tol):

    ##Prepare input data
    #Convert fa to radians
    #fa_rad=fa * np.pi/180.
    
    #We need an array of integers to tell the function which points are for the baseline.  This will be 1:BLpts
    BL_idx=np.arange(1,BLpts,1)

    #We need to set the signal model to be spgr which will require the tr, the flip angle and a dummy te (ignored by default)
    sigmod=spgr(tr, fa, 1)
    #We also need to set the c_to_r_model to be linear, with relaxivity r1 and dummy r2 relaxivity
    conc_to_r=c_to_r_linear(r1,0)

    
    # run test
    #The code uses two functions to get from SI to conc
    enh_curve=sig_to_enh(s_array, BL_idx)

    #B1 correction factor k=1

    conc_curve = enh_to_conc(enh_curve, 1, 1/T1base, conc_to_r, sigmod)


    np.testing.assert_allclose( conc_curve, conc_array, rtol=r_tol, atol=a_tol )

