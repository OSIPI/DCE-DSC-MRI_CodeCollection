import os

import numpy as np
import pandas as pd

# Summary of tolerances, starting values and bounds (where fitting is implented
# within the tests):
# ve: a_tol=0.05, r_tol=0, start=0.2, bounds=(0,1)
# PS/KTrans: a_tol=0.005, r_tol=0.1, start=0.6, bounds=(0,5), units /min
# vp: a_tol=0.025, r_tol=0, start=0.01, bounds=(0,1)
# fp: a_tol=5, r_tol=0.1, start=20, bounds=(0,200) , units ml/100ml/min
# E: start=0.15, bounds=(0,1)
# delay: a_tol=0.2, r_tol=0, start=0, bounds=(-10,10), units s


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
    a_tol_vp = [0.025] * len(Ktrans_ref)
    a_tol_ve = [0.05] * len(Ktrans_ref)
    a_tol_Ktrans = [0.005] * len(Ktrans_ref)
    a_tol_delay = [0.2] * len(Ktrans_ref)
    r_tol_vp = [0] * len(Ktrans_ref)
    r_tol_ve = [0] * len(Ktrans_ref)
    r_tol_Ktrans = [0.1] * len(Ktrans_ref)
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
    r_tol_Ktrans = [0.1] * len(Ktrans_ref)
    r_tol_delay = [0] * len(Ktrans_ref)

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(
        zip(label, t_array, C_array, ca_array, ta_array, ve_ref, Ktrans_ref, arterial_delay_ref, a_tol_ve, r_tol_ve,
            a_tol_Ktrans, r_tol_Ktrans, a_tol_delay, r_tol_delay))

    return pars


def dce_DRO_data_Patlak():
    """
    Import dce concentration data for testing.

    Data summary: simulated Patlak model data
    Patient(s): n.a.
    Source: Concentration-time data generated by M. Thrippleton using Matlab
        code at https://github.com/mjt320/DCE-functions
    Detailed info:
        Temporal resolution: 0.5 s
        Acquisition time: 300 s
        AIF: Parker function, starting at t=10s
        Noise: SD = 0.02 mM
        Arterial delay: none
    Reference values:
        Reference values are the parameters used to generate the data.
        All combinations of vp (0.1, 0.2, 0.5) and PS (0, 5, 15)*1e-2 per min
        are included.
    Citation:
        Code used in Manning et al., Magnetic Resonance in Medicine, 2021
        https://doi.org/10.1002/mrm.28833
        Matlab code: https://github.com/mjt320/DCE-functions

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case
    """
    filename = os.path.join(os.path.dirname(__file__), 'data',
                            'patlak_sd_0.02_delay_0.csv')

    # read from CSV to pandas
    converters = {'t': lambda x: np.fromstring(x, dtype=float, sep=' '),
                  'C_t': lambda x: np.fromstring(x, dtype=float, sep=' '),
                  'cp_aif': lambda x: np.fromstring(x, dtype=float, sep=' '),
                  }
    df = pd.read_csv(filename, converters=converters)

    # convert to lists
    label = df['label'].tolist()  # label describing entry
    t_array = df['t'].tolist()  # seconds
    C_t_array = df['C_t'].tolist()  # mM
    cp_aif_array = df['cp_aif'].tolist()  # mM
    vp_ref = df['vp'].tolist()
    ps_ref = df['ps'].tolist()  # per min

    # set the tolerance to use for this dataset
    a_tol_vp = [0.025] * len(vp_ref)
    a_tol_ps = [0.005] * len(vp_ref)
    r_tol_vp = [0] * len(vp_ref)
    r_tol_ps = [0.1] * len(vp_ref)

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(
        zip(label, t_array, C_t_array, cp_aif_array, vp_ref, ps_ref, a_tol_vp,
            r_tol_vp, a_tol_ps, r_tol_ps))

    return pars


def dce_DRO_data_2cxm():
    """
    Import dce concentration data for testing.

    Data summary: simulated 2CXM model data
    Patient(s): n.a.
    Source: Concentration-time data generated by M. Thrippleton using Matlab
        code at https://github.com/mjt320/DCE-functions
    Detailed info:
        Temporal resolution: 0.5 s
        Acquisition time: 300 s
        AIF: Parker function, starting at t=10s
        Noise: SD = 0.001 mM
        Arterial delay: none
        Since it is challenging to fit all parameters across a wide area of
        parameter space, data is generated with high SNR.
    Reference values:
        Reference values are the parameters used to generate the data.
        All combinations of the following are included:
         vp = [0.02, 0.1]
         ve = [0.1, 0.2]
         fp = [5, 25, 40] 100ml/ml/min
         ps_range = [0.05, 0.15] per min
    Citation:
        Code used in Manning et al., Magnetic Resonance in Medicine, 2021
        https://doi.org/10.1002/mrm.28833
        Matlab code: https://github.com/mjt320/DCE-functions

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case
    """
    filename = os.path.join(os.path.dirname(__file__), 'data',
                            '2cxm_sd_0.001_delay_0.csv')

    # read from CSV to pandas
    converters = {'t': lambda x: np.fromstring(x, dtype=float, sep=' '),
                  'C_t': lambda x: np.fromstring(x, dtype=float, sep=' '),
                  'cp_aif': lambda x: np.fromstring(x, dtype=float, sep=' '),
                  }
    df = pd.read_csv(filename, converters=converters)
    # convert to lists
    label = df['label'].tolist()  # label describing entry
    t_array = df['t'].tolist()  # seconds
    C_t_array = df['C_t'].tolist()  # mM
    cp_aif_array = df['cp_aif'].tolist()  # mM
    vp_ref = df['vp'].tolist()
    ve_ref = df['ve'].tolist()
    fp_ref = df['fp'].tolist()  # ml/100ml/min
    ps_ref = df['ps'].tolist()  # /min

    # set the tolerance to use for this dataset
    a_tol_vp = [0.025] * len(vp_ref)
    a_tol_ve = [0.05] * len(vp_ref)
    a_tol_fp = [5] * len(vp_ref)
    a_tol_ps = [0.005] * len(vp_ref)
    r_tol_vp = [0] * len(vp_ref)
    r_tol_ve = [0] * len(vp_ref)
    r_tol_fp = [0.1] * len(vp_ref)
    r_tol_ps = [0.1] * len(vp_ref)

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(
        zip(label, t_array, C_t_array, cp_aif_array, vp_ref, ve_ref, fp_ref,
            ps_ref, a_tol_vp, r_tol_vp, a_tol_ve, r_tol_ve, a_tol_fp,
            r_tol_fp, a_tol_ps, r_tol_ps))

    return pars


def dce_DRO_data_2cum():
    """
    Import dce concentration data for testing.

    Data summary: simulated 2CUM model data
    Patient(s): n.a.
    Source: Concentration-time data generated by M. Thrippleton using Matlab
        code at https://github.com/mjt320/DCE-functions
    Detailed info:
        Temporal resolution: 0.5 s
        Acquisition time: 300 s
        AIF: Parker function, starting at t=10s
        Noise: SD = 0.02 mM
        Arterial delay: none
        Note: data are genearted using the 2CXM model with ve=100
    Reference values:
        Reference values are the parameters used to generate the data.
        All combinations of the following are included:
            vp_range = [0.02, 0.05, 0.1]; % dimensionless
            fp_range = [5, 25, 40]; % 100ml/ml/min
            ps_range = [1e-5 0.01, 0.025]; % per min
    Citation:
        Code used in Manning et al., Magnetic Resonance in Medicine, 2021
        https://doi.org/10.1002/mrm.28833
        Matlab code: https://github.com/mjt320/DCE-functions

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case
    """
    filename = os.path.join(os.path.dirname(__file__), 'data',
                            '2cum_sd_0.02_delay_0.csv')

    # read from CSV to pandas
    converters = {'t': lambda x: np.fromstring(x, dtype=float, sep=' '),
                  'C_t': lambda x: np.fromstring(x, dtype=float, sep=' '),
                  'cp_aif': lambda x: np.fromstring(x, dtype=float, sep=' '),
                  }
    df = pd.read_csv(filename, converters=converters)
    # convert to lists
    label = df['label'].tolist()  # label describing entry
    t_array = df['t'].tolist()  # seconds
    C_t_array = df['C_t'].tolist()  # mM
    cp_aif_array = df['cp_aif'].tolist()  # mM
    vp_ref = df['vp'].tolist()
    fp_ref = df['fp'].tolist()  # 100ml/ml/min
    ps_ref = df['ps'].tolist()  # /min

    # set the tolerance to use for this dataset
    a_tol_vp = [0.025] * len(vp_ref)
    a_tol_fp = [5] * len(vp_ref)
    a_tol_ps = [0.005] * len(vp_ref)
    r_tol_vp = [0] * len(vp_ref)
    r_tol_fp = [0.1] * len(vp_ref)
    r_tol_ps = [0.1] * len(vp_ref)

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(
        zip(label, t_array, C_t_array, cp_aif_array, vp_ref, fp_ref,
            ps_ref, a_tol_vp, r_tol_vp, a_tol_fp,
            r_tol_fp, a_tol_ps, r_tol_ps))

    return pars
