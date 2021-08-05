import os

import numpy as np
import pandas as pd

# Functions to return formatted test data for testing T1 mapping functionality
# 1 function is defined per test dataset



def dce_test_data():
    """
    Import dce concentration data for testing.
    
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
    Tolerance: 0.05 /s + 0.05 (relative)


    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case

    """    
    filename = os.path.join(os.path.dirname(__file__), 'data', 'dce_test_data.csv')    
    # read from CSV to pandas
    converters = {
        'C': lambda x:np.fromstring(x, dtype=float, sep=' '),
        't': lambda x:np.fromstring(x, dtype=float, sep=' '),
        'ca': lambda x:np.fromstring(x, dtype=float, sep=' ')
        }              
    df = pd.read_csv(filename, converters = converters)
    
    # convert to lists
    label = df['label'].tolist() # label describing entry
    t_array = df['t'].tolist() # seconds
    C_array = df['C'].tolist() # mM
    ca_array = df['s'].tolist() # mM
    ta_ref = df['ta'].tolist() # seconds
    ve_ref = df['ve'].tolist()
    vp_ref = df['vp'].tolist()
    Ktrans_ref = df['Ktrans'].tolist()	
    
    # set the tolerance to use for this dataset
    a_tol = [0.05] * len(Ktrans_ref)
    r_tol = [0.05] * len(Ktrans_ref)
    
    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(zip(label, t_array, C_array, ca_array, ta_ref, ve_ref, vp_ref, Ktrans_ref, a_tol, r_tol))
    
    return pars



# def t1_quiba_data():
#     """
#     Import variable flip angle T1 data for testing.
    
#     Data summary: simulated T1 mapping data
#     Patient(s): digital reference object (DRO)
#     Source: QIBA T1 DRO v3
#     Detailed info: The DRO consists of multiple groups of 10x10 voxels, each with
#         a different combination of of noise level, S0 and R1. 1 voxel is selected
#         per combination and  voxels with S0/sigma < 1500 are excluded.        
#     Reference values: R1 reference values are thosed used to generate the data.
#     Citation: Daniel Barboriak, https://qidw.rsna.org/#collection/594810551cac0a4ec8ffe574
#     Comments: Low SNR voxels are excluded since we cannot expect any code to find
#         the ground-truth value.
#     Tolerance: 0.05 /s + 0.05 (relative)


#     Returns
#     -------
#     pars : list of tuples
#         Input for pytest.mark.parametrize
#         Each tuple contains a set of parameters corresponding to 1 test case

#     """   
#     filename = os.path.join(os.path.dirname(__file__), 'data', 't1_quiba_data.csv')    
    
#     # read from CSV to pandas
#     converters = {
#         's': lambda x:np.fromstring(x, dtype=float, sep=' '),
#         'FA': lambda x:np.fromstring(x, dtype=float, sep=' '),
#         'TR': lambda x:np.fromstring(x, dtype=float, sep=' ')
#         }              
#     df = pd.read_csv(filename, converters = converters)
    
#     # convert to lists
#     label = df['label'].tolist() # label describing entry
#     fa_array = df['FA'].tolist() # degrees
#     tr_array = df['TR'].tolist() # s
#     s_array = df['s'].tolist()
#     r1_ref = (df['R1']*1000.).tolist() # convert /ms to /s
#     s0_ref = df['s0'].tolist()
 
#     # set the tolerance to use for this dataset
#     a_tol = [0.05] * len(s0_ref)
#     r_tol = [0.05] * len(s0_ref)
    
#     # convert to list of tuples (input for pytest.mark.parametrize)
#     pars = list(zip(label, fa_array, tr_array, s_array, r1_ref, s0_ref, a_tol, r_tol))
    
#     return pars



# def t1_prostate_data():
#     """
#     Import variable flip angle T1 data for testing.
    
#     Data summary: in-vivo prostate 3-T T1 mapping data
#     Patient(s): 5 patients with prostate cancer
#     Source: department of Radiation Oncology, the Netherlands Cancer Institute,
#         drTHERAPAT study
#     Detailed info: each entry corresponds to a randomly selected voxel in the 
#         prostate from variable flip angle SPGR images.
#     Reference values: T1 reference values obtained using in-house Matlab code
#     Citation: Klawer et al., " Improved repeatability of dynamic contrast-enhanced 
#         MRI using the complex MRI signal to derive arterial input functions: 
#         a test-retest study in prostate cancer patients." Magn Reson Med, 2019.
#     Comments: T1 values are provided with and without B1 correction and both
#         with linear and nonlinear fitting procedures.
#         Currently, we only test non-B1-corrected data and use reference values
#         based on non-linear fitting.
#     Tolerance: 0.05 /s + 0.05 (relative)

#     Returns
#     -------
#     pars : list of tuples
#         Input for pytest.mark.parametrize
#         Each tuple contains a set of parameters corresponding to 1 test case

#     """   
#     filename = os.path.join(os.path.dirname(__file__), 'data', 't1_prostate_data.csv')    
    
#     # read from CSV to pandas
#     converters = {
#         's': lambda x:np.fromstring(x, dtype=float, sep=' '),
#         'FA': lambda x:np.fromstring(x, dtype=float, sep=' '),
#         'TR': lambda x:np.fromstring(x, dtype=float, sep=' ')
#         }              
#     df = pd.read_csv(filename, converters = converters)
    
#     # convert to lists
#     label = df['label'].tolist() # label describing entry
#     fa_array = df['FA'].tolist() # degrees
#     tr_array = (df['TR']/1000).tolist() # s
#     s_array = df['s'].tolist()
#     r1_ref = (1000/df[' T1 nonlinear']).tolist() # convert T1 (ms) to R1 (/s)
#     s0_ref = df[' s0 nonlinear'].tolist()
 
#     # set the tolerance to use for this dataset
#     a_tol = [0.05] * len(s0_ref)
#     r_tol = [0.05] * len(s0_ref)
    
#     # convert to list of tuples (input for pytest.mark.parametrize)
#     pars = list(zip(label, fa_array, tr_array, s_array, r1_ref, s0_ref, a_tol, r_tol))
    
#     return pars
    
   

    