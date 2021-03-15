import pytest
import numpy as np
import pandas as pd

# Functions to return test for testing T1 mapping functionality
# 1 function is defined per dataset



def t1_brain_data():
    """
    Import variable flip angle T1 data for testing
    Data summary: in-vivo 3-T T1 mapping data
    Patient(s): mild-stroke patient
    Source: University of Edinburgh, Mild Stroke Study 3
    Detailed info: each entry corresponds to a single voxel following spatial realignment of variable flip angle images, take from ROIs drawn in the white matter
    Reference values: obtained using in-house matlab code https://github.com/mjt320/HIFI
    Citation: Clancy, U., et al., "Rationale and design of a longitudinal study of cerebral small vessel diseases, clinical and imaging outcomes in patients presenting with mild ischaemic stroke: Mild Stroke Study 3." European Stroke Journal, 2020.
    Comments: reference T1 values are not B1-corrected, thus do not reflect true T1.


    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple is the set of values for 1 test

    """
    
    filename = 'test/T1_mapping/data/t1_brain_data.csv'
    
    # read from CSV to pandas
    converters = {
        's': lambda x:np.fromstring(x, dtype=float, sep=' '),
        'FA': lambda x:np.fromstring(x, dtype=float, sep=' '),
        'TR': lambda x:np.fromstring(x, dtype=float, sep=' ')
        }              
    df = pd.read_csv(filename, converters = converters)
    
    # convert to lists
    label = df['label'].tolist() # label describing entry
    fa_array = df['FA'].tolist() # degrees
    tr_array = df['TR'].tolist() # s
    s_array = df['s'].tolist()
    r1_ref = df['R1'].tolist() # /s
    s0_ref = df['s0'].tolist()
    
    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(zip(label, fa_array, tr_array, s_array, r1_ref, s0_ref))
    
    return pars


def t1_quiba_data():
    #TODO: document
    """
    Import variable flip angle T1 data for testing
    Data summary: ADD INFO
    Patient(s): 
    Source: 
    Detailed info: 
    Reference values: 
    Citation: 
    Comments: 


    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple is the set of values for 1 test

    """
    
    filename = 'test/T1_mapping/data/t1_quiba_data.csv'
    
    # read from CSV to pandas
    converters = {
        's': lambda x:np.fromstring(x, dtype=float, sep=' '),
        'FA': lambda x:np.fromstring(x, dtype=float, sep=' '),
        'TR': lambda x:np.fromstring(x, dtype=float, sep=' ')
        }              
    df = pd.read_csv(filename, converters = converters)
    
    # convert to lists
    label = df['label'].tolist() # label describing entry
    fa_array = df['FA'].tolist() # degrees
    tr_array = df['TR'].tolist() # s
    s_array = df['s'].tolist()
    r1_ref = (df['R1']*1000.).tolist() # /s
    s0_ref = df['s0'].tolist()
 
    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(zip(label, fa_array, tr_array, s_array, r1_ref, s0_ref))
    
    return pars

def t1_prostate_data():
    # same idea...
    return []

    
   
# combine all test data to decorate test functions    
parameters = pytest.mark.parametrize('label, fa_array, tr_array, s_array, r1_ref, s0_ref',
                     t1_brain_data() +
                     t1_quiba_data() +
                     t1_prostate_data()                     
                     )
    