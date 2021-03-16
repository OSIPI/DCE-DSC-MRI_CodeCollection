import os

import numpy as np
import pandas as pd

# Functions to return formatted test data for testing T1 mapping functionality
# 1 function is defined per test dataset



def t1_brain_data():
    """
    Import variable flip angle T1 data for testing.
    
    Data summary: in-vivo brain 3-T T1 mapping data
    Patient(s): 1 mild-stroke patient
    Source: University of Edinburgh, Mild Stroke Study 3
    Detailed info: each entry corresponds to a voxel following spatial realignment
        of variable flip angle SPGR images, taken from ROIs drawn in the white matter,
        deep gray matter and cerebrospinal fluid.
    Reference values: R1 reference values obtained using in-house Matlab code
        (https://github.com/mjt320/HIFI)
    Citation: Clancy, U., et al., "Rationale and design of a longitudinal study
        of cerebral small vessel diseases, clinical and imaging outcomes in patients
        presenting with mild ischaemic stroke: Mild Stroke Study 3." European Stroke
        Journal, 2020.
    Comments: R1 reference values are not B1-corrected here, thus may not reflect
        true R1.


    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case

    """    
    filename = os.path.join(os.path.dirname(__file__), 'data', 't1_brain_data.csv')    
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
    """
    Import variable flip angle T1 data for testing.
    
    Data summary: simulated T1 mapping data
    Patient(s): digital reference object (DRO)
    Source: QIBA T1 DRO v3
    Detailed info: The DRO consists of multiple groups of 10x10 voxels, each with
        a different combination of of noise level, S0 and R1. 1 voxel is selected
        per combination and only voxels with SNR < 30 are excluded.        
    Reference values: R1 reference values are thosed used to generate the data.
    Citation: Daniel Barboriak, https://qidw.rsna.org/#collection/594810551cac0a4ec8ffe574
    Comments: Low SNR voxels are excluded since we cannot expect any code to find
        the ground-truth value.


    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case

    """   
    filename = os.path.join(os.path.dirname(__file__), 'data', 't1_quiba_data.csv')    
    
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

    
   

    