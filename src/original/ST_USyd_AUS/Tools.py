#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 12:08:02 2018

@author: Sirisha Tadimalla
"""

# Import libraries
import numpy as np

#####################################
# Shifts array to the right by n elements
# and inserts n zeros at the beginning of the array


def arr_shift(A, n):
    shift = np.zeros(n)
    A_shifted = np.insert(A, 0, shift)
    A_new = A_shifted[0 : len(A)]
    return A_new


#####################################
# Performs convolution of (1/T)exp(-t/T) with a


def expconv(T, t, a):
    if T == 0:
        return a

    n = len(t)
    f = np.zeros((n,))

    x = (t[1 : n - 1] - t[0 : n - 2]) / T
    da = (a[1 : n - 1] - a[0 : n - 2]) / x

    E = np.exp(-x)
    E0 = 1 - E
    E1 = x - E0

    add = a[0 : n - 2] * E0 + da * E1

    for i in range(0, n - 2):
        f[i + 1] = E[i] * f[i] + add[i]

    f[n - 1] = f[n - 2]
    return f


#####################################
# Performs deconvolution of C and ca_time where
# ca_time = ca times dt


def deconvolve(C, ca, dt):
    # Build matrix from given AIF
    ca_time = ca * dt
    A = np.zeros(shape=(len(ca), len(ca)))
    for i in np.arange(0, len(ca)):
        A[:, i] = arr_shift(ca_time, i)

    # SVD of A
    U, S, Vt = np.linalg.svd(A, full_matrices=True)

    # Inverse of A
    cutoff = 0.01
    S[S < cutoff * np.max(S)] = 0
    nz = S != 0
    S[nz] = 1 / S[nz]

    invA = np.matmul(np.transpose(Vt), np.matmul(np.diag(S), np.transpose(U)))

    # Solution
    X = np.matmul(invA, C)

    return X


#####################################
# Performs discrete integration of ca
# time t


def integrate(ca, t):
    f = np.zeros(len(ca))
    dt = t[1] - t[0]
    f[0] = 0
    for n in np.arange(1, len(t)):
        f[n] = dt * ca[n] + f[n - 1]

    return f


#####################################
# Calculates the SPGR signal using given
# flip angle (FA in degrees), repetition time (TR in seconds),
# equilibrium signal (S0) and longitudinal
# relaxation rate (R1 in Hz)


def spgress(FA, TR, S0, R1):
    E = np.exp(-TR * R1)
    c = np.cos(np.array(FA) * np.pi / 180)
    s = np.sin(np.array(FA) * np.pi / 180)
    Mz = np.absolute(S0 * s * (1 - E) / (1 - c * E))
    return Mz


#####################################
# Calculates the post-contrast longitudinal relaxation rate (R1) from the
# dynamic SPGR signal S, given the flip angle (FA in degrees),
# repetition time (TR in seconds),
# equilibrium signal (S0), precontrast longitudinal
# relaxation rate (R10 in Hz)


def spgress_inv(S, FA, TR, S0, R10):
    E = np.exp(-TR * R10)
    c = np.cos(np.array(FA) * np.pi / 180)
    Sn = (S / S0) * (1 - E) / (1 - c * E)  # normalised signal
    R1 = -np.log((1 - Sn) / (1 - c * Sn)) / TR  # Relaxation rate in 1/s
    return R1
