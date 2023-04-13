import numpy as np
import os
from .. import math


def loadFromFile(filename):
    # returns (time, AIF)
    return np.loadtxt(filename, skiprows=1, delimiter=";", unpack=True)


def loadStandard(a=None, b=None, i=None):
    # returns a standard, high res AIF.
    # if a and b are set, it returns the AIF at position /Data/AIF_DCE/a/b/AIF_DCE.txt
    # a and b are ignored if i is set. If i is set, then all the AIF's available are
    # aranged in a list, and the i'th AIF is selected
    # returns (time, AIF)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if i is not None:
        AIFFiles = []
        for root, dirs, files in os.walk(dir_path + "/../Data/AIF_DCE/"):
            if len(files) > 0:
                if files[0][-4:] == ".txt":
                    AIFFiles.append(root + "/" + files[0])
        if i > len(AIFFiles) - 1:
            print("i is too large. Should be <= {}.".format(len(AIFFiles) - 1))
        C_a = np.loadtxt(AIFFiles[i])
        t = np.linspace(0, 600, len(C_a))
    elif not None in [a, b]:
        C_a = np.loadtxt(dir_path + "/../Data/AIF_DCE/" + a + "/" + b + "/AIF_DCE.txt")
        t = np.linspace(0, 600, len(C_a))
    else:
        t, C_a = loadFromFile(dir_path + "/../Data/Aorta.txt")

    return t, C_a


def gammaVariate(t, t_0=None, A=None, alpha=None, beta=None, y_max=None, t_max=None):
    """
    There are two call signatures to this function.
    Either all of [t, t_0, alpha, beta] must be set,
    or all of [t, alpha, y_max, t_max].

    The reason for this distinction is that alpha and
    beta are dependent variables. All dependencies
    are eliminated in the second way of calling
    this function [1].


    [1] Madsen, Mark T. "A simplified formulation of the gamma variate function." Physics in Medicine and Biology 37.7 (1992): 1597.
    """

    if not None in [t_0, alpha, beta]:
        return A * (t - t_0) ** alpha * np.exp(-(t - t_0) / beta)
    elif not None in [alpha, y_max, t_max]:
        return (
            y_max
            * t_max ** (-alpha)
            * np.exp(alpha)
            * t**alpha
            * np.exp(-alpha * t / t_max)
        )
    else:
        raise TypeError("Missing one or more arguments.")


def gammaVariateFit(t, y):
    # start with a random, reasonable guess for the paramaters
    y_max_guess_mid = np.max(y)
    t_max_guess_mid = t[np.argmax(y)]

    y_max_guess_max = y_max_guess_mid * 1.1
    y_max_guess_min = y_max_guess_mid * 0.9
    y_max_guess = np.linspace(y_max_guess_min, y_max_guess_max, 20)

    t_max_guess_max = t_max_guess_mid * 1.1
    t_max_guess_min = t_max_guess_mid * 0.9
    t_max_guess = np.linspace(t_max_guess_min, t_max_guess_max, 20)

    alpha_guess_max = 10
    alpha_guess_min = 0
    alpha_guess = np.linspace(alpha_guess_min, alpha_guess_max, 20)

    best = np.sum(
        np.abs(
            gammaVariate(
                t, alpha=alpha_guess[0], y_max=y_max_guess[0], t_max=t_max_guess[0]
            )
            - y
        )
        ** 2
    )
    alpha_best = alpha_guess[0]
    t_max_best = t_max_guess[0]
    y_max_best = y_max_guess[0]

    found = False
    while not found:
        for y_max in y_max_guess:
            for t_max in t_max_guess:
                for alpha in alpha_guess:
                    this_diff = np.sum(
                        np.abs(
                            gammaVariate(t, alpha=alpha, y_max=y_max, t_max=t_max) - y
                        )
                        ** 2
                    )
                    if this_diff < best:
                        best = this_diff
                        alpha_best = alpha
                        t_max_best = t_max
                        y_max_best = y_max

        found = True
        if abs(alpha_best - alpha_guess_max) < 1e-4:
            alpha_guess_min = alpha_guess_max
            alpha_guess_max *= 1.1
            found = False
        elif abs(alpha_best - alpha_guess_min) < 1e-4:
            alpha_guess_max = alpha_guess_min
            alpha_guess_min *= 0.9
            found = False

        if abs(t_max_best - t_max_guess_max) < 1e-4:
            t_max_guess_min = t_max_guess_max
            t_max_guess_max *= 1.1
            found = False
        elif abs(t_max_best - t_max_guess_min) < 1e-4:
            t_max_guess_max = t_max_guess_min
            t_max_guess_min *= 0.9
            found = False

        if abs(y_max_best - y_max_guess_max) < 1e-4:
            y_max_guess_min = y_max_guess_max
            y_max_guess_max *= 1.1
            found = False
        elif abs(y_max_best - y_max_guess_min) < 1e-4:
            y_max_guess_max = y_max_guess_min
            y_max_guess_min *= 0.9
            found = False

        alpha_guess = np.linspace(alpha_guess_min, alpha_guess_max, 20)
        t_max_guess = np.linspace(t_max_guess_min, t_max_guess_max, 20)
        y_max_guess = np.linspace(y_max_guess_min, y_max_guess_max, 20)

    return gammaVariate(t, alpha=alpha_best, t_max=t_max_best, y_max=y_max_best)


def modify(t, C_a, TTP=100, FWHM=10, height=6):
    currmax = np.max(C_a) / 2.0
    diff = abs(C_a - currmax)

    first = np.argmin(diff)
    diff[first] = np.max(diff)

    second = np.argmin(diff)

    currFWHM = abs(t[first] - t[second])

    t2 = t * FWHM / currFWHM

    C_a = C_a * height / np.max(C_a)

    currTTPidx = np.argmax(C_a)
    TTPidx = np.argmin(np.abs(t2 - TTP))
    TTPdiff = currTTPidx - TTPidx

    if TTPdiff > 0:
        C_a = np.delete(C_a, np.arange(TTPdiff))
    if TTPdiff < 0:
        C_a = np.hstack([np.zeros(np.abs(TTPdiff)), C_a])

    t_max = np.max(t)
    t2 = t2[: np.argmin(abs(t2 - t_max))]
    C_a = C_a[: len(t2)]

    return t2, C_a
