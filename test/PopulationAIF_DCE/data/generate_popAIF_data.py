import os
import pandas as pd
import numpy as np
import math

def generate_GeorgiouAIF_refdata():
    """
    This function imports and creates the test data to test implementations of the Georgiou AIF.

    The test data is based on the original data submitted with the manuscript (mrm27524-sup-0002-figs2.xlsx) from
    Georgiou et al. MRM 2018. This data is labeled as 'Original_AIF'
    instead of creating a csv file with the data, the original data is imported and permutations of this original file
    are created within this function. This includes AIFs with different temporal resolutions.

    not included in test data yet:
    - time arrays longer than 5 min
    - bolus arrival time variations

    References: Georgiou et al. MRM 2018, doi: 10.1002/mrm.27524

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case
    """

    label = []
    time = []
    cb_ref_values = []

    label.append('original_AIF')
    # import xsl file
    filename_original_aif = os.path.join(os.path.dirname(__file__), 'mrm27524-sup-0002-figs2.xlsx')
    original_data_xls = pd.read_excel(filename_original_aif)
    time_original = original_data_xls["time (min)"].to_numpy()
    values_original = original_data_xls["[Gd-DOTA] (mM)"].to_numpy()
    time.append(time_original)
    cb_ref_values.append(values_original)

    # create for loop with different permutations of temporal resolution (original is at 0.1 s)
    range_dt = np.array([0.5, 1, 2, 2.5, 5, 7.5])  #range of realistic temporal resolutions
    for current_dt in range_dt:
        current_label = 'temp_res_' + str(current_dt) + 's'
        label.append(current_label)
        current_dt_min = current_dt / 60
        time_int = np.arange(0, 5, current_dt_min)
        cb = np.interp(time_int, time_original, values_original)
        time.append(time_int)
        cb_ref_values.append(cb)

    # write to csv file: label, time, cb_ref_values
    ref_values = []
    for lb, tm, ref in zip(label, time, cb_ref_values):
        for t, r in zip(tm, ref):
            ref_values.append([lb, t, r])
    filename_ref = os.path.join(os.path.dirname(__file__), '', 'GeorgiouAIF_ref.csv')
    pd.DataFrame(data=ref_values, columns=["label", "time", "Cb"]).to_csv(filename_ref, index=False)

def ParkerAIF_refdata():
    """
    This function imports and creates the test data to test implementations of the Parker AIF.

    For the Parker AIF the parameters of the functional form are copied from Table 1 of the reference

    These parameters in combination with equation 1 were used to create different versions of the ParkerAIF.
    The original data had a temp resolution of 4.97 s and tot acquisition time of 372.75 s. This data was labeled as 'Original_AIF'
    The other reference entries include AIF values with varying temporal resolutions and varying total acquisition times.
    Data without a delay or pre-contrast value are assumed. Those are included in a separate testset

    Bolus arrival time variations are included in ParkerAIF_refdata_delay

    References: Parker et al. MRM 2006, doi: 10.1002/mrm.21066

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case
    """

    # AIF parameters from Table 1 of Parker et al.
    A1 = 0.809
    sigma1 = 0.0563
    T1 = 0.17046
    A2 = 0.330
    sigma2 = 0.132
    T2 = 0.365
    alpha = 1.05
    beta = 0.1685
    s = 38.078
    tau = 0.483

    label = []
    time = []
    cb_ref_values = []
    delay = []

    label.append('original_AIF')
    time_original = np.arange(0, 5, 4.97/60) # in min
    gaussian1 = A1 / (sigma1 * np.sqrt(2 * np.pi)) * np.exp(-np.square(time_original - T1) / (2 * np.square(sigma1)))
    gaussian2 = A2 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(-np.square(time_original - T2) / (2 * np.square(sigma2)))
    modSigm = (alpha * np.exp(-beta * time_original)) / (1 + np.exp(-s * (time_original - tau)))
    cb = np.add(gaussian1, gaussian2)
    cb = np.add(cb, modSigm)
    time.append(time_original)
    cb_ref_values.append(cb)
    delay.append(0)  # assume no delay

    # similar for a range of temporal resolutions
    range_dt = np.array([0.5, 1, 2, 2.5, 5, 7.5])  # range of realistic temporal resolutions
    for current_dt in range_dt:
        current_label = 'temp_res_' + str(current_dt) + 's'
        label.append(current_label)
        current_dt_min = current_dt / 60
        time_int = np.arange(0, 5, current_dt_min)
        gaussian1 = A1 / (sigma1 * np.sqrt(2 * np.pi)) * np.exp(-np.square(time_int - T1) / (2 * np.square(sigma1)))
        gaussian2 = A2 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(-np.square(time_int - T2) / (2 * np.square(sigma2)))
        modSigm = (alpha * np.exp(-beta * time_int)) / (1 + np.exp(-s * (time_int - tau)))
        cb = np.add(gaussian1, gaussian2)
        cb = np.add(cb, modSigm)
        time.append(time_int)
        cb_ref_values.append(cb)
        delay.append(0)  # assume no delay

    # similar for different acquisition times
    range_endt = np.array([3, 5, 7, 10])  # range of total acquisition time
    current_dt_min = 2.5 / 60  # current temp resolution 2.5 s
    for current_endt in range_endt:
        current_label = 'acq_time_' + str(current_endt) + 'min'
        label.append(current_label)
        time_int = np.arange(0, current_endt, current_dt_min)
        gaussian1 = A1 / (sigma1 * np.sqrt(2 * np.pi)) * np.exp(-np.square(time_int - T1) / (2 * np.square(sigma1)))
        gaussian2 = A2 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(-np.square(time_int - T2) / (2 * np.square(sigma2)))
        modSigm = (alpha * np.exp(-beta * time_int)) / (1 + np.exp(-s * (time_int - tau)))
        cb = np.add(gaussian1, gaussian2)
        cb = np.add(cb, modSigm)
        time.append(time_int)
        cb_ref_values.append(cb)
        delay.append(0)  # assume no delay

    # set the tolerance to use for this dataset
    a_tol = [0.05] * len(label)
    r_tol = [0.05] * len(label)

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(zip(label, time, cb_ref_values, delay, a_tol, r_tol))

    return pars


def ParkerAIF_refdata_delay():
    """
    This function creates the test data to test implementations of the Parker AIF including a delay or precontrast signal.

    For the Parker AIF the parameters of the functional form are copied from Table 1 of the reference

    In the function ParkerAIF_refdataa() no delay was assumed. In this case the original AIF was used as a starting point (temp resol 4.97s, tot acquisition time 5 min);
    This was extended with delays of exact multiplications of the temporal resolution (4.97s) as well as random seconds

    References: Parker et al. MRM 2006, doi: 10.1002/mrm.21066

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case
    """

    # AIF parameters from Table 1 of Parker et al.
    A1 = 0.809
    sigma1 = 0.0563
    T1 = 0.17046
    A2 = 0.330
    sigma2 = 0.132
    T2 = 0.365
    alpha = 1.05
    beta = 0.1685
    s = 38.078
    tau = 0.483

    label = []
    time = []
    cb_ref_values = []
    delay = []

    dt = 4.97  # temp resolution in seconds
    time_current = np.arange(0, 5, dt/60)  # in min

    # similar for a range of temporal resolutions
    delay_range = np.array([0, dt, 2*dt, 5*dt, 2, 5, 10, 18, 31])  # indicated in seconds
    for current_delay in delay_range:
        current_label = 'delay_' + str(current_delay) + 's'
        label.append(current_label)
        gaussian1 = A1 / (sigma1 * np.sqrt(2 * np.pi)) * np.exp(-np.square(time_current - T1) / (2 * np.square(sigma1)))
        gaussian2 = A2 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(-np.square(time_current - T2) / (2 * np.square(sigma2)))
        modSigm = (alpha * np.exp(-beta * time_current)) / (1 + np.exp(-s * (time_current - tau)))
        cb = np.add(gaussian1, gaussian2)
        cb = np.add(cb, modSigm)
        # shift cb according to delay value
        n_shift = math.floor(current_delay/dt)
        cb = np.pad(cb, (n_shift, 0), 'constant')
        cb = cb[0:len(time_current)]  # keep array length constant
        time.append(time_current)
        cb_ref_values.append(cb)
        delay.append(current_delay)  # assume no delay


    # set the tolerance to use for this dataset
    a_tol = [0.05] * len(label)
    r_tol = [0.05] * len(label)

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(zip(label, time, cb_ref_values, delay, a_tol, r_tol))

    return pars

generate_GeorgiouAIF_refdata()