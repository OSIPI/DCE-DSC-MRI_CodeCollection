"""Utilities.

Created 6 October 2021
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Functions:
    minimize_global
    least_squares_global
"""

import nibabel as nib
import numpy as np
from scipy.optimize import minimize, least_squares


def minimize_global(cost, x_0_all, **minimizer_kwargs):
    """Find global minimum by calling scipy.optimize.minimize multiple times.

    Parameters
    ----------
    cost : function
        Function to be minimised.
    x_0_all : list
        list of 1D ndarrays. Each contains a set of initial parameter values.
    **minimizer_kwargs : optional keyword arguments accepted by minimize

    Returns
    -------
    result : OptimizeResult
        OptimizeResult corresponding to the fitting attempt with the lowest
        minimum.

    """
    results = [minimize(cost, x_0, **minimizer_kwargs) for x_0 in x_0_all]
    costs = [result.fun if result.success is False else np.nan
             for result in results]
    cost = min(costs)
    idx = costs.index(cost)
    result = results[idx]

    return result


def least_squares_global(res, x_0_all, **least_squares_kwargs):
    """Find global minimum using scipy.optimize.least_squares multiple times.

    Parameters
    ----------
    res : function
          Function that generates vector of residuals. See scipy docs.
    x_0_all : list
        list of 1D ndarrays. Each contains a set of initial parameter values.
    **least_squares_kwargs : optional keyword arguments accepted by minimize

    Returns
    -------
    result : OptimizeResult
        OptimizeResult corresponding to the fitting attempt with the lowest
        minimum.

    """

    results = [least_squares(res, x_0, **least_squares_kwargs) for x_0 in x_0_all]
    costs = [result.cost if result.success is False else np.nan
             for result in results]
    cost = min(costs)
    idx = costs.index(cost)
    result = results[idx]

    return result
