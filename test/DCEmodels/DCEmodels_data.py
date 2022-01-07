import os

import numpy as np
import pandas as pd


def dce_DRO_data_extended_tofts_kety(delay=False):
    """
    Import dce concentration data for testing.

    Data summary: digital reference object of the brain.
    Patient(s): n.a.
    Source: Creating an anthropomorphic digital MR phantom—an extensible tool
        for comparing and evaluating quantitative imaging algorithms. PMB 2016.
        RJ Bosca, EF Jackson: https://iopscience.iop.org/article/10.1088/0031
        -9155/61/2/974 https://qidw.rsna.org/#collection/594810551cac0a4ec8ffe574
        /folder/5e20ccb8b3467a6a9210e9ff MR Modality Datasets / Dynamic Contrast
        Enhanced (DCE) MRI / DCE-MRI DRO Data and Code / DCE-MRI DRO Data -
        Developmental Anthropomorphic (Ryan Bosca / Ed Jackson)
    Detailed info:
        Test case labels: test_vox_T{tumour voxel number}_{SNR},
        e.g. test_vox_T1_30
        Selected voxels:
            datVIF=alldata[108,121,6,:]
            datT1=alldata[121,87,6,:] --> tumour voxel 1
            datT2=alldata[156,105,6,:] --> tumour voxel 2
            datT3=alldata[139,93,6,:] --> tumour voxel 3
        These are signal values, which were converted to concentration curves
        using dce_to_r1eff from
        https://github.com/welcheb/pydcemri/blob/master from David S. Smith
    Reference values: reference values were found from the accompanying pdf
        document, which describes the values per voxel. T1 blood of 1440; T1
        tissue of 1084 for WM, 1820 for GM, 1000 for T1-T3; TR=5 ms; FA=30;
        Hct=0.45
    Citation: Bosca, Ryan J., and Edward F. Jackson. "Creating an
        anthropomorphic digital MR phantom—an extensible tool for comparing and
        evaluating quantitative imaging algorithms." Physics in Medicine &
        Biology 61.2 (2016): 974.

    Parameters
    ----------
    delay : Bool
        If True, a delay of 5 time points is applied to the tissue
        concentration curve. Used to test code that fits a delay. Defaults to
        False.

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case

    """
    filename = os.path.join(os.path.dirname(__file__), 'data', 'dce_DRO_data_extended_tofts.csv')
    # read from CSV to pandas
    converters = {'C': lambda x: np.fromstring(x, dtype=float, sep=' '),
                  't': lambda x: np.fromstring(x, dtype=float, sep=' '),
                  'ca': lambda x: np.fromstring(x, dtype=float, sep=' '),
                  'ta': lambda x: np.fromstring(x, dtype=float, sep=' '), }
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
    if delay:  # delay tissue curve by 5 time points
        C_array = np.array(C_array)
        C_array = C_array[:, :-5]
        C_array = np.concatenate((np.tile([0], [len(C_array), 5]), C_array), axis=1)
        arterial_delay_ref = np.array(arterial_delay_ref) + t_array[0][5]
        for a in range(len(label)):
            label[a] = label[a] + '_delayed'

    # set the tolerance to use for this dataset
    a_tol_ve = [0.05] * len(Ktrans_ref)
    a_tol_vp = [0.0025] * len(Ktrans_ref)
    a_tol_Ktrans = [0.005] * len(Ktrans_ref)
    a_tol_delay = [0.2] * len(Ktrans_ref)
    r_tol_ve = [0] * len(Ktrans_ref)
    r_tol_vp = [0] * len(Ktrans_ref)
    r_tol_Ktrans = [0.01] * len(Ktrans_ref)
    r_tol_delay = [0] * len(Ktrans_ref)

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(
        zip(label, t_array, C_array, ca_array, ta_array, ve_ref, vp_ref, Ktrans_ref, arterial_delay_ref, a_tol_ve,
            r_tol_ve, a_tol_vp, r_tol_vp, a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay))

    return pars


def dce_DRO_data_tofts(delay=False):
    """
    Import dce concentration data for testing.

    Data summary: QIBA digital reference object.
    Patient(s): n.a.
    Source: https://qibawiki.rsna.org/images/1/14/QIBA_DRO_2015_v1.42.pdf and
        https://qidw.rsna.org/#collection/594810551cac0a4ec8ffe574/folder/5e20ccb8b3467a6a9210e9ff
        in folder: MR Modality Datasets / Dynamic Contrast Enhanced (DCE) MRI
            / DCE-MRI DRO Data and Code / DCE-MRI DRO Data (Daniel Barboriak) /
            QIBA_v11_Tofts / QIBA_v11_Tofts_GE / T1_tissue_500 / DICOM_dyn
    Detailed info: digital reference object
        Test case labels: test_vox_T{parameter combination}_{SNR},
            e.g. test_vox_T5_30
        To get the high SNR dataset, data was averaged as follows:
            datVIF = data[:, :9, :]
            datVIF = np.mean(datVIF, axis=(0, 1))
            datT1 = np.mean(data[44:49, 13:18, :], (0, 1))
            datT2 = np.mean(data[32:37, 21:26, :], (0, 1))
            datT3 = np.mean(data[42:47, 23:28, :], (0, 1))
            datT4 = np.mean(data[22:27, 32:37, :], (0, 1))
            datT5 = np.mean(data[22:27, 43:48, :], (0, 1))
        Noise was added to the high-SNR data to obtain data at different SNRs
        these are signal values, which were converted to concentration curves
        using dce_to_r1eff from
        https://github.com/welcheb/pydcemri/blob/master from David S. Smith
    Reference values: reference values were found from the accompanying pdf
        document, which describes the values per voxel. T1 blood of 1440;
        T1 tissue of 500; TR=5 ms; FA=30;Hct=0.45
    Citation: QIBA Algorithm Comparison Activities: Digital Reference Objects
        and Software Evaluation

    Parameters
    ----------
    delay : Bool
        If True, a delay of 5 time points is applied to the tissue
        concentration curve. Used to test code that fits a delay. Defaults to
        False.

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case
    """
    filename = os.path.join(os.path.dirname(__file__), 'data', 'dce_DRO_data_tofts.csv')
    # read from CSV to pandas
    converters = {'C': lambda x: np.fromstring(x, dtype=float, sep=' '),
                  't': lambda x: np.fromstring(x, dtype=float, sep=' '),
                  'ca': lambda x: np.fromstring(x, dtype=float, sep=' '),
                  'ta': lambda x: np.fromstring(x, dtype=float, sep=' '), }
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
    if delay:  # delay tissue curve by 5 time points
        C_array = np.array(C_array)
        C_array = C_array[:, :-5]
        C_array = np.concatenate((np.tile([0], [len(C_array), 5]), C_array), axis=1)
        arterial_delay_ref = np.array(arterial_delay_ref) + t_array[0][5] / 60
        for a in range(len(label)):
            label[a] = label[a] + '_delayed'

    # set the tolerance to use for this dataset
    a_tol_ve = [0.05] * len(Ktrans_ref)
    a_tol_Ktrans = [0.005] * len(Ktrans_ref)
    a_tol_delay = [0.2] * len(Ktrans_ref)
    r_tol_ve = [0] * len(Ktrans_ref)
    r_tol_Ktrans = [0.01] * len(Ktrans_ref)
    r_tol_delay = [0] * len(Ktrans_ref)

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(
        zip(label, t_array, C_array, ca_array, ta_array, ve_ref, Ktrans_ref, arterial_delay_ref, a_tol_ve, r_tol_ve,
            a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay))

    return pars
