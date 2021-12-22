import os

import numpy as np
import pandas as pd


def dce_DRO_data_extended_tofts_kety(delay=False):
    """
    Import dce concentration data for testing.

    Data summary: digital reference object of the brain.
    Patient(s): n.a.
    Source: Creating an anthropomorphic digital MR phantom—an extensible tool for comparing and evaluating quantitative imaging algorithms. PMB 2016. RJ Bosca, EF Jackson: https://iopscience.iop.org/article/10.1088/0031-9155/61/2/974
    https://qidw.rsna.org/#collection/594810551cac0a4ec8ffe574/folder/5e20ccb8b3467a6a9210e9ff
    MR Modality Datasets / Dynamic Contrast Enhanced (DCE) MRI / DCE-MRI DRO Data and Code / DCE-MRI DRO Data - Developmental Anthropomorphic (Ryan Bosca / Ed Jackson)
    Detailed info:

    Selected voxels:
    datVIF=alldata[108,121,6,:]
    datWM=alldata[86,97,6,:] --> white matter
    datGM=alldata[54,136,6,:] --> gray matter
    datT1=alldata[121,87,6,:] --> tumour voxel 1
    datT2=alldata[156,105,6,:] --> tumour voxel 2
    datT3=alldata[139,93,6,:] --> tumour voxel 3

    these are signal values, which were converted to concentration curves using dce_to_r1eff from
    https://github.com/welcheb/pydcemri/blob/master from David S. Smith

    Reference values: reference values were found from the accompanying pdf document, which describes the values per voxel. T1 blood of 1440; T1 tissue of 1084 for WM, 1820 for GM, 1000 for T1-T3; TR=5 ms; FA=30;Hct=0.45

    Reference values: Reference values obtained from their Mat files
    Citation: Bosca, Ryan J., and Edward F. Jackson. "Creating an anthropomorphic digital MR phantom—an extensible tool for comparing and evaluating quantitative imaging algorithms." Physics in Medicine & Biology 61.2 (2016): 974.

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case

    """
    filename = os.path.join(os.path.dirname(__file__), 'data', 'dce_DRO_data_extended_tofts.csv')
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
        arterial_delay_ref = np.array(arterial_delay_ref) + t_array[0][5]
        for a in range(len(label)):
            label[a] = label[a] + '_delayed'

    # set the tolerance to use for this dataset
    # set the tolerance to use for this dataset
    a_tol_ve = [0.01] * int(len(Ktrans_ref) / 5) + [0.04] * int(len(Ktrans_ref) / 5) + [0.03] * int(len(Ktrans_ref) / 5) + [0.02] * int(len(Ktrans_ref) / 5) + [0.01] * int(len(Ktrans_ref) / 5)# absolute tolerance
    a_tol_vp = [0.002] * int(len(Ktrans_ref) / 5) + [0.008] * int(len(Ktrans_ref) / 5) + [0.006] * int(len(Ktrans_ref) / 5) + [0.004] * int(len(Ktrans_ref) / 5) + [0.002] * int(len(Ktrans_ref) / 5)# absolute tolerance
    a_tol_Ktrans = [0.005] * int(len(Ktrans_ref) / 5) + [0.02] * int(len(Ktrans_ref) / 5) + [0.015] * int(len(Ktrans_ref) / 5) + [0.01] * int(len(Ktrans_ref) / 5) + [0.005] * int(len(Ktrans_ref) / 5)# absolute tolerance
    a_tol_delay = [0.2] * len(Ktrans_ref)  # absolute tolerance
    r_tol_ve = [0.025] * int(len(Ktrans_ref) / 5) + [0.1] * int(len(Ktrans_ref) / 5) + [0.075] * int(len(Ktrans_ref) / 5) + [0.05] * int(len(Ktrans_ref) / 5) + [0.025] * int(len(Ktrans_ref) / 5)  # relative tolerance
    r_tol_vp = [0.025] * int(len(Ktrans_ref) / 5) + [0.1] * int(len(Ktrans_ref) / 5) + [0.075] * int(len(Ktrans_ref) / 5) + [0.05] * int(len(Ktrans_ref) / 5) + [0.025] * int(len(Ktrans_ref) / 5)  # relative tolerance
    r_tol_Ktrans = [0.025] * int(len(Ktrans_ref) / 5) + [0.1] * int(len(Ktrans_ref) / 5) + [0.075] * int(len(Ktrans_ref) / 5) + [0.05] * int(len(Ktrans_ref) / 5) + [0.025] * int(len(Ktrans_ref) / 5)  # relative tolerance
    r_tol_delay = [0.1] * len(Ktrans_ref)  # relative tolerance

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(
        zip(label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref, a_tol_ve,
            r_tol_ve, a_tol_vp, r_tol_vp, a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay))

    return pars



def dce_DRO_data_tofts(delay=False):
    """
    Import dce concentration data for testing.

    Data summary: digital reference object.
    Patient(s): n.a.
    Source: https://qibawiki.rsna.org/images/1/14/QIBA_DRO_2015_v1.42.pdf and https://qidw.rsna.org/#collection/594810551cac0a4ec8ffe574/folder/5e20ccb8b3467a6a9210e9ff
    in folder:
    MR Modality Datasets / Dynamic Contrast Enhanced (DCE) MRI / DCE-MRI DRO Data and Code / DCE-MRI DRO Data (Daniel Barboriak) / QIBA_v11_Tofts / QIBA_v11_Tofts_GE / T1_tissue_500 / DICOM_dyn

    Detailed info: digital reference object

    To get the high SNR dataset, data was averaged as follows:
    datVIF = data[:, :9, :]
    datVIF = np.mean(datVIF, axis=(0, 1))
    datT1 = np.mean(data[44:49, 13:18, :], (0, 1))
    datT2 = np.mean(data[32:37, 21:26, :], (0, 1))
    datT3 = np.mean(data[42:47, 23:28, :], (0, 1))
    datT4 = np.mean(data[22:27, 32:37, :], (0, 1))
    datT5 = np.mean(data[22:27, 43:48, :], (0, 1))

    these are signal values, which were converted to concentration curves using dce_to_r1eff from
    https://github.com/welcheb/pydcemri/blob/master from David S. Smith

    Reference values: reference values were found from the accompanying pdf document, which describes the values per voxel. T1 blood of 1440; T1 tissue of 500; TR=5 ms; FA=30;Hct=0.45
        Citation: QIBA Algorithm Comparison Activities: Digital Reference Objects and Software Evaluation

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
    a_tol_ve = [0.01] * int(len(Ktrans_ref) / 5) + [0.04] * int(len(Ktrans_ref) / 5) + [0.03] * int(len(Ktrans_ref) / 5) + [0.02] * int(len(Ktrans_ref) / 5) + [0.01] * int(len(Ktrans_ref) / 5)# absolute tolerance
    a_tol_Ktrans = [0.005] * int(len(Ktrans_ref) / 5) + [0.02] * int(len(Ktrans_ref) / 5) + [0.015] * int(len(Ktrans_ref) / 5) + [0.01] * int(len(Ktrans_ref) / 5) + [0.005] * int(len(Ktrans_ref) / 5)# absolute tolerance
    a_tol_delay = [0.2] * len(Ktrans_ref)  # absolute tolerance
    r_tol_ve = [0.025] * int(len(Ktrans_ref) / 5) + [0.1] * int(len(Ktrans_ref) / 5) + [0.075] * int(len(Ktrans_ref) / 5) + [0.05] * int(len(Ktrans_ref) / 5) + [0.025] * int(len(Ktrans_ref) / 5)  # relative tolerance
    r_tol_Ktrans = [0.05] * int(len(Ktrans_ref) / 5) + [0.1] * int(len(Ktrans_ref) / 5) + [0.075] * int(len(Ktrans_ref) / 5) + [0.05] * int(len(Ktrans_ref) / 5) + [0.05] * int(len(Ktrans_ref) / 5)  # relative tolerance
    r_tol_delay = [0.1] * len(Ktrans_ref)  # relative tolerance

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(
        zip(label, t_array, C_array, ca_array, ta_array, ve_ref, Ktrans_ref, arterial_delay_ref, a_tol_ve,
            r_tol_ve, a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay))

    return pars
