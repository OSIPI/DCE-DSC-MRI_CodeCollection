import os
import pandas as pd
import numpy as np
import math


def GeorgiouAIF_refdata():
    """
    This function imports the test data to test implementations of the Georgiou AIF.

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

    # set the tolerances to use for this dataset
    a_tol = 0.0001
    r_tol = 0.01

    # load csv file
    filename = os.path.join(os.path.dirname(__file__), "data", "GeorgiouAIF_ref.csv")
    df = pd.read_csv(filename)
    df_label = df.groupby("label")
    pars = []
    # convert to list as input for pytest.mark.parametrize
    for current_label, values in df_label:
        new_list = (
            current_label,
            values.time.to_numpy(),
            values.Cb.to_numpy(),
            a_tol,
            r_tol,
        )
        pars.append(new_list)

    return pars


def ParkerAIF_refdata():
    """
    This function imports the test data to test implementations of the Parker AIF.

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

    # set the tolerances to use for this dataset
    a_tol = 0.0001
    r_tol = 0.01

    # load csv file
    filename = os.path.join(os.path.dirname(__file__), "data", "ParkerAIF_ref.csv")
    df = pd.read_csv(filename)
    df_label = df.groupby("label")
    pars = []
    # convert to list as input for pytest.mark.parametrize
    for current_label, values in df_label:
        new_list = (
            current_label,
            values.time.to_numpy(),
            values.Cb.to_numpy(),
            values.delay.to_numpy()[0],
            a_tol,
            r_tol,
        )
        pars.append(new_list)

    return pars


def ParkerAIF_refdata_delay():
    """
    This function imports the test data to test implementations of the Parker AIF including a delay or pre-contrast signal.

    For the Parker AIF the parameters of the functional form are copied from Table 1 of the reference

    References: Parker et al. MRM 2006, doi: 10.1002/mrm.21066

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case
    """

    # set the tolerances to use for this dataset
    a_tol = 0.1  # the tolerances are set differently for the non-delay data as due to the approach for resampling there are some differences between implementations.
    r_tol = 0.1

    # load csv file
    filename = os.path.join(
        os.path.dirname(__file__), "data", "ParkerAIF_ref_with_delay.csv"
    )
    df = pd.read_csv(filename)
    df_label = df.groupby("label")
    pars = []
    # convert to list as input for pytest.mark.parametrize
    for current_label, values in df_label:
        new_list = (
            current_label,
            values.time.to_numpy(),
            values.Cb.to_numpy(),
            values.delay.to_numpy()[0],
            a_tol,
            r_tol,
        )
        pars.append(new_list)

    return pars


def preclinical_refdata():
    """
    This function imports the test data to test implementations of the preclinical AIF of McGrath et al.

    To create the preclinical AIF the parameters of the functional form are copied from Table 1, model B of the reference

    These parameters in combination with equation 5 were used to create different versions of the preclinicalAIF.
    The original data had a temp resolution of 0.5s and tot acquisition time of 300s. This data was labeled as 'Original_AIF'
    The other reference entries include AIF values with varying temporal resolutions and varying total acquisition times.
    Data without a delay or pre-contrast value are assumed. Those are included in a separate testset

    Bolus arrival time variations are included in preclinicalAIF_refdata_delay

    References: McGrath et al. MRM 2009, DOI: 10.1002/mrm.21959

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case
    """

    # set the tolerances to use for this dataset
    a_tol = 0.0001
    r_tol = 0.01

    # load csv file
    filename = os.path.join(os.path.dirname(__file__), "data", "preclinicalAIF_ref.csv")
    df = pd.read_csv(filename)
    df_label = df.groupby("label")
    pars = []
    # convert to list as input for pytest.mark.parametrize
    for current_label, values in df_label:
        new_list = (
            current_label,
            values.time.to_numpy(),
            values.Cb.to_numpy(),
            values.delay.to_numpy()[0],
            a_tol,
            r_tol,
        )
        pars.append(new_list)

    return pars


def preclinical_refdata_delay():
    """
    This function imports the test data to test implementations of the preclinical AIF of McGrath et al. including a delay
    for more details see preclinical_refdata()

    To create the preclinical AIF the parameters of the functional form are copied from Table 1, model B of the reference

    References: McGrath et al. MRM 2009, DOI: 10.1002/mrm.21959

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case
    """

    # set the tolerances to use for this dataset
    a_tol = 0.0001
    r_tol = 0.01

    # load csv file
    filename = os.path.join(
        os.path.dirname(__file__), "data", "preclinicalAIF_ref_delay.csv"
    )
    df = pd.read_csv(filename)
    df_label = df.groupby("label")
    pars = []
    # convert to list as input for pytest.mark.parametrize
    for current_label, values in df_label:
        new_list = (
            current_label,
            values.time.to_numpy(),
            values.Cb.to_numpy(),
            values.delay.to_numpy()[0],
            a_tol,
            r_tol,
        )
        pars.append(new_list)

    return pars
