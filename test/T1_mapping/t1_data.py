import pytest
import numpy as np
import pandas as pd

# Functions to return test for testing T1 mapping functionality
# 1 function is defined per dataset

def t1_brain_data():
    """
    Import variable flip angle T1 data for testing
    Data summary: in-vivo 3-T T1 mapping data
    Patient(s): voxels from single white matter ROI in 1 patient
    Reference values: fitted using TBC
    Citation: TBC    

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple is the set of values for 1 test

    """
    
    filename = './data/mask_WM_data.csv'
    
    # read from CSV to pandas
    converters = {
        's': lambda x:np.fromstring(x, dtype=float, sep=' '),
        'FA': lambda x:np.fromstring(x, dtype=float, sep=' '),
        'TR': lambda x:np.fromstring(x, dtype=float, sep=' ')
        }              
    df = pd.read_csv(filename, converters = converters)
    
    # convert to lists
    fa_array = df['FA'].tolist()
    tr_array = df['TR'].tolist()
    s_array = df['s'].tolist()
    r1_ref = df['R1'].tolist()
    s0_ref = df['s0'].tolist()
    
    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(zip(fa_array, tr_array, s_array, r1_ref, s0_ref))
    
    return pars

def t1_prostate_data():
    # same idea...
    return []
    
def t1_some_more_data():
    return []
    # as above