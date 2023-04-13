"""Functions for DSC processing.

Created 15 October 2021
@authors: Laura Bell
@institution: Barrow Neurological Institute

Translated from jupyter notebook DSCpipeline.ipynb by Michael Thrippleton

Functions:
    estimate_R2s
    estimate_delta_R2s
    estimate_delta_R2s_dual_echo
"""

import numpy as np


def estimate_R2s(s1, s2, te1, te2):
    """Estimate R2* and TE=0 signal from dual echo signals.

    Parameters
    ----------
    s1 : float
        Signal (first echo).
    s2 : float
        Signal (second echo).
    te1 : float
        First echo time (s).
    te2 : float
        Second echo time (s).

    Returns
    -------
    s_te0 : float
        Signal at zero echo time.
    R2s : float
        R2* (s^-1).

    """
    R2s = (1 / (te2 - te1)) * np.log(s1 / s2)
    s_te0 = s1 * np.exp(te1 * R2s)
    return s_te0, R2s


def estimate_delta_R2s(s, basepts_range, te):
    """Estimate R2* change for single echo data.

    Parameters
    ----------
    s : ndarray
        1D array of signals.
    basepts_range : list
        2-element list indicating [first, last] index to use for baseline
        signal estimation.
    te : float
        Echo time (s).

    Returns
    -------
    delta_R2s : ndarray
        1D array of R2* changes (s^-1).

    """
    delta_R2s = (
        -1 / te * (np.log(s / np.mean(s[basepts_range[0] : basepts_range[1] + 1])))
    )
    return delta_R2s


def estimate_delta_R2s_dual_echo(s1, s2, basepts_range, te1, te2):
    """Estimate R2* change for dual echo data.

    Parameters
    ----------
    s1 : ndarray
        1D array of first-echo signals.
    s2 : ndarray
        1D array of second-echo signals.
    basepts_range : list
        2-element list indicating [first, last] index to use for baseline
        signal estimation.
    te1 : float
        First echo time (s).
    te2 : float
        Second echo time (s).

    Returns
    -------
    delta_R2s : ndarray
        1D array of R2* changes (s^-1).

    """
    s1_base = np.mean(s1[basepts_range[0] : basepts_range[1] + 1])
    s2_base = np.mean(s2[basepts_range[0] : basepts_range[1] + 1])
    delta_R2s = 1 / (te2 - te1) * (np.log((s1 / s1_base) / (s2 / s2_base)))
    return delta_R2s
