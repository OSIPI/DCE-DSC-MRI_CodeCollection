import os

import numpy as np
import pandas as pd

# Functions to return formatted test data for testing SI to concentration code

def SI2Conc_data():
    """
    Import signal intensity data for testing conversion to concentration
    
    Data summary: Signal intensity curves derived from 1 patient 
    Patient(s): Randomly selected patient DCE-MRI of uterus 
    Source: University of Edinburgh, Mechanistic substudy of UCON https://www.birmingham.ac.uk/research/bctu/trials/womens/ucon/ucon-home.aspx used with permission.
    Detailed info: Each entry corresponds to signal intensity curve from voxels in uterus, or aorta.  They were converted to concentration
    using code from the University of Edinburgh (same as that used in Reavey, J.J., Walker, C., Nicol, M., Murray, A.A., Critchley, H.O.D., Kershaw, L.E., Maybin, J.A., 2021. 
    Markers of human endometrial hypoxia can be detected in vivo and ex vivo during physiological menstruation. Hum. Reprod. 36, 941–950.)
    but with various flip angles, baseline T1 values etc rather than the actual values used in the acquisition, to test a wider range of possibilites.

    Data file lines consist of; label, flip angle, TR, baseline T1, number of baseline points before contrast, r1, signal, concentration

    Tolerance: 0.00001 + 0.00001 (relative) - this is just maths so it should be very similar for the different implementations
   
    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case
    """    
    filename = os.path.join(os.path.dirname(__file__), 'data', 'SI2Conc_data.csv')    
    # read from CSV to pandas
    converters = {
        's': lambda x:np.fromstring(x, dtype=float, sep=' '),
        'conc': lambda x:np.fromstring(x, dtype=float, sep=' '),
        }              
    df = pd.read_csv(filename, converters = converters)
    
    # convert to lists
    label = df['label'].tolist() # label describing entry
    fa = df['FA'].tolist() # degrees
    tr = df['TR'].tolist() # s
    T1base = df['T1base'].tolist() # s
    BLpts = df['numbaselinepts'].tolist()
    r1 = df['r1'].tolist()
    s_array = df['s'].tolist()
    conc_array = df['conc'].tolist()
    
    # set the tolerance to use for this dataset
    a_tol = [0.00001] * len(s_array)
    r_tol = [0.00001] * len(s_array)
    
    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(zip(label, fa, tr, T1base, BLpts, r1, s_array, conc_array, a_tol, r_tol))
    
    return pars



