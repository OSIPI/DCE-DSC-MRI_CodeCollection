import os
from xmlrpc.client import boolean

import numpy as np
import pandas as pd
import scipy.io as sio


def dsc_DRO_data_vascular_model(delay=False):
    """
    Import dsc concentration data for testing.

    Data summary: digital reference object consisting of signal time curves
    representing perfusion scenarios typical of grey and white matter.

    Source: https://github.com/arthur-chakwizira/BezierCurveDeconvolution
    Ref: Non-parametric deconvolution using Bézier curves for quantification
    of cerebral perfusion in dynamic susceptibility contrast MRI

    Similar digital reference objects have previously been used in publications
    evaluating perfusion deconvolution/estimation techniques including Wu et al. 2003
    (DOI 10.1002/mrm.10522), Mouridsen et al. 2006 (DOI 10.1016/j.neuroimage.2006.06.015),
    Chappell et al. 2015 (10.1002/mrm.25390)

    Parameters
    ----------
    Transit time distribution: gamma variate distribution
    with shape parameter lambda = 3

    delay : Bool
        Not applied yet

    Returns
    -------
    pars : list of tuples
        Input for pytest.mark.parametrize
        Each tuple contains a set of parameters corresponding to 1 test case

    """

    filename = os.path.join(os.path.dirname(__file__), "data", "dsc_data.csv")
    # read from CSV to pandas
    converters = {
        "C_tis": lambda x: np.fromstring(x, dtype=float, sep=" "),
        "C_aif": lambda x: np.fromstring(x, dtype=float, sep=" "),
    }
    df = pd.read_csv(filename, converters=converters)

    label = df["label"].tolist()  # label describing entry
    C_tis = df["C_tis"].tolist()
    C_aif = df["C_aif"].tolist()
    cbv = df["cbv"].tolist()  # ml/100ml
    cbf = df["cbf"].tolist()  # ml/100ml/min
    tr = df["tr"].tolist()  # seconds

    # set the tolerance to use for this dataset
    r_tol_cbv = [0.1] * len(tr)
    r_tol_cbf = [0.1] * len(tr)
    a_tol_cbv = [1] * len(tr)
    a_tol_cbf = [15] * len(tr)

    # convert to list of tuples (input for pytest.mark.parametrize)
    pars = list(
        zip(
            label,
            C_tis,
            C_aif,
            tr,
            cbv,
            cbf,
            r_tol_cbv,
            r_tol_cbf,
            a_tol_cbv,
            a_tol_cbf,
        )
    )

    return pars
