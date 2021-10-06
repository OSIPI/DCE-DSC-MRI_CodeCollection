import os
import pandas as pd
import numpy as np

def GeorgiouAIF_refdata():
    """
    This function imports the test data to test the implementation of the Georgiou AIF.

    The test data is based on the original data submitted with the manuscript (mrm27524-sup-0002-figs2.xlsx) from
    Georgiou et al. MRM 2018. This data is labeled as 'Original_AIF'
    instead of creating a csv file with the data, the original data is imported and permutations of this original file
    are created within this function. This includes AIFs with different temporal resolutions.

    not included in test data yet:
    - time arrays larger than 5 min!
    - bolus arrival time variations

    References: Geourgiou et al. MRM 2018, doi: 10.1002/mrm.27524

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case
    """

    #filename = os.path.join(os.path.dirname(__file__), 'data', 'GeorgiouAIF_testdata.csv')
    # read from CSV to pandas
    #df = pd.read_csv(filename)

    label = []
    time = []
    cb_ref_values = []

    label.append('original_AIF')
    # import xsl file
    original_data_xls = pd.read_excel('mrm27524-sup-0002-figs2.xlsx')
    time_original = original_data_xls["time (min)"].to_numpy()
    values_original = original_data_xls["[Gd-DOTA] (mM)"].to_numpy()
    time.append(time_original)
    cb_ref_values.append(values_original)

    # first permutation with a dt = 2 s
    label.append('temp_res_2s')
    dt = 2 / 60
    time_int = np.arange(0, 5, dt)
    cb = np.interp(time_int, time_original, values_original)
    time.append(time_int)
    cb_ref_values.append(cb)

    # set the tolerance to use for this dataset
    a_tol = [0.05] * len(label)
    r_tol = [0.05] * len(label)

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(zip(label, time, cb_ref_values, a_tol, r_tol))

    return pars