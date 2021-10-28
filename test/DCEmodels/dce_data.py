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
    filename = os.path.join(os.path.dirname(__file__), 'data', 'dce_prostate_data.csv')    
    # read from CSV to pandas
    converters = {
        'C': lambda x:np.fromstring(x, dtype=float, sep=' '),
        't': lambda x:np.fromstring(x, dtype=float, sep=' '),
        'ca': lambda x:np.fromstring(x, dtype=float, sep=' '),
        'ta': lambda x: np.fromstring(x, dtype=float, sep=' '),
        }
    df = pd.read_csv(filename, converters = converters)
    
    # convert to lists
    label = df['label'].tolist() # label describing entry
    t_array = df['t'].tolist() # seconds
    C_array = df['C'].tolist() # mM
    ca_array = df['ca'].tolist() # mM
    ta_array = df['ta'].tolist() # seconds
    ve_ref = df['ve'].tolist()
    vp_ref = df['vp'].tolist()
    Ktrans_ref = df['Ktrans'].tolist()	
    arterial_delay_ref = df['arterialdelay'].tolist()

    # set the tolerance to use for this dataset
    a_tol_ve = [0.1] * len(Ktrans_ref) #absolute tolerance
    a_tol_vp = [0.01] * len(Ktrans_ref)  # absolute tolerance
    a_tol_Ktrans = [0.5] * len(Ktrans_ref)  # absolute tolerance
    a_tol_delay = [0.2] * len(Ktrans_ref)  # absolute tolerance
    r_tol_ve = [0.10] * len(Ktrans_ref) #relative tolerance
    r_tol_vp = [0.10] * len(Ktrans_ref) #relative tolerance
    r_tol_Ktrans = [0.10] * len(Ktrans_ref) #relative tolerance
    r_tol_delay = [0.10] * len(Ktrans_ref) #relative tolerance

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(zip(label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref, a_tol_ve, r_tol_ve, a_tol_vp,r_tol_vp,a_tol_Ktrans,r_tol_Ktrans,a_tol_delay,r_tol_delay))
    
    return pars


def dce_DRO_data(delay=False):
    """
    Import dce concentration data for testing.

    Data summary: digital reference object of the brain.
    Patient(s): n.a.
    Source: Creating an anthropomorphic digital MR phantom—an extensible tool for comparing and evaluating quantitative imaging algorithms. PMB 2016. RJ Bosca, EF Jackson: https://iopscience.iop.org/article/10.1088/0031-9155/61/2/974
    Detailed info:
    Reference values: R1 reference values obtained from their Mat files
    Citation: Bosca, Ryan J., and Edward F. Jackson. "Creating an anthropomorphic digital MR phantom—an extensible tool for comparing and evaluating quantitative imaging algorithms." Physics in Medicine & Biology 61.2 (2016): 974.
    Comments: the signal values were converted to concentration curves using dce_to_r1eff from https://github.com/welcheb/pydcemri/blob/master from David S. Smith
    Tolerance: ? /s + ? (relative)


    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case

    """
    filename = os.path.join(os.path.dirname(__file__), 'data', 'dce_DRO_data.csv')
    # read from CSV to pandas
    converters = {
        'C': lambda x: np.fromstring(x, dtype=float, sep=' '),
        't': lambda x: np.fromstring(x, dtype=float, sep=' '),
        'ca': lambda x: np.fromstring(x, dtype=float, sep=' '),
        'ta': lambda x: np.fromstring(x, dtype=float, sep=' '),
    }
    df = pd.read_csv(filename, converters=converters)

    # convert to lists
    label = df['label'].tolist()  # label describing entry
    t_array = df['t'].tolist()  # seconds
    C_array = df['C'].tolist()  # mM
    ca_array = df['ca'].tolist()  # mM
    ta_array = df['ta'].tolist()  # seconds
    ve_ref = df['ve'].tolist()
    vp_ref = df['vp'].tolist()
    Ktrans_ref = df['Ktrans'].tolist()
    arterial_delay_ref = df['arterialdelay'].tolist()
    if delay:
        C_array=np.array(C_array)
        C_array=C_array[:,:-5]
        C_array=np.concatenate((np.tile([0],[len(C_array),5]),C_array),axis=1)
        #t_array = np.array(t_array) + 15
        arterial_delay_ref = np.array(arterial_delay_ref) + t_array[0][5]/60
        for a in range(len(label)):
            label[a] = label[a] + '_delayed'

    # set the tolerance to use for this dataset
    # set the tolerance to use for this dataset
    a_tol_ve = [0.01] * len(Ktrans_ref)  # absolute tolerance
    a_tol_vp = [0.001] * len(Ktrans_ref)  # absolute tolerance
    a_tol_Ktrans = [0.0075] * len(Ktrans_ref)  # absolute tolerance
    a_tol_delay = [0.01] * len(Ktrans_ref)  # absolute tolerance
    r_tol_ve = [0.10] * len(Ktrans_ref)  # relative tolerance
    r_tol_vp = [0.10] * len(Ktrans_ref)  # relative tolerance
    r_tol_Ktrans = [0.10] * len(Ktrans_ref)  # relative tolerance
    r_tol_delay = [0.10] * len(Ktrans_ref)  # relative tolerance

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(
        zip(label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref, a_tol_ve,
            r_tol_ve, a_tol_vp, r_tol_vp, a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay))

    return pars



def dce_DRO_data_tofts(delay=False):
    """
    Import dce concentration data for testing.

    Data summary: digital reference object of the brain.
    Patient(s): n.a.
    Source: Creating an anthropomorphic digital MR phantom—an extensible tool for comparing and evaluating quantitative imaging algorithms. PMB 2016. RJ Bosca, EF Jackson: https://iopscience.iop.org/article/10.1088/0031-9155/61/2/974
    Detailed info:
    Reference values: R1 reference values obtained from their Mat files
    Citation: Bosca, Ryan J., and Edward F. Jackson. "Creating an anthropomorphic digital MR phantom—an extensible tool for comparing and evaluating quantitative imaging algorithms." Physics in Medicine & Biology 61.2 (2016): 974.
    Comments: the signal values were converted to concentration curves using dce_to_r1eff from https://github.com/welcheb/pydcemri/blob/master from David S. Smith
    Tolerance: ? /s + ? (relative)


    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case

    """
    filename = os.path.join(os.path.dirname(__file__), 'data', 'dce_DRO_data_tofts.csv')
    # read from CSV to pandas
    converters = {
        'C': lambda x: np.fromstring(x, dtype=float, sep=' '),
        't': lambda x: np.fromstring(x, dtype=float, sep=' '),
        'ca': lambda x: np.fromstring(x, dtype=float, sep=' '),
        'ta': lambda x: np.fromstring(x, dtype=float, sep=' '),
    }
    df = pd.read_csv(filename, converters=converters)

    # convert to lists
    label = df['label'].tolist()  # label describing entry
    t_array = df['t'].tolist()  # seconds
    C_array = df['C'].tolist()  # mM
    ca_array = df['ca'].tolist()  # mM
    ta_array = df['ta'].tolist()  # seconds
    ve_ref = df['ve'].tolist()
    Ktrans_ref = df['Ktrans'].tolist()
    arterial_delay_ref = df['arterialdelay'].tolist()
    if delay:
        C_array=np.array(C_array)
        C_array=C_array[:,:-5]
        C_array=np.concatenate((np.tile([0],[len(C_array),5]),C_array),axis=1)
        #t_array = np.array(t_array) + 15
        arterial_delay_ref = np.array(arterial_delay_ref) + t_array[0][5]/60
        for a in range(len(label)):
            label[a] = label[a] + '_delayed'

    # set the tolerance to use for this dataset
    # set the tolerance to use for this dataset
    a_tol_ve = [0.01] * len(Ktrans_ref)  # absolute tolerance
    a_tol_Ktrans = [0.0075] * len(Ktrans_ref)  # absolute tolerance
    a_tol_delay = [0.03] * len(Ktrans_ref)  # absolute tolerance
    r_tol_ve = [0.10] * len(Ktrans_ref)  # relative tolerance
    r_tol_Ktrans = [0.10] * len(Ktrans_ref)  # relative tolerance
    r_tol_delay = [0.10] * len(Ktrans_ref)  # relative tolerance

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(
        zip(label, t_array, C_array, ca_array, ta_array, ve_ref, Ktrans_ref, arterial_delay_ref, a_tol_ve,
            r_tol_ve, a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay))

    return pars
