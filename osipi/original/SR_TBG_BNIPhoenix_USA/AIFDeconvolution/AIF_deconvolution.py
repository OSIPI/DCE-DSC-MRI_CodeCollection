#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 05:17:08 2021

authors: Sudarshan Ragunathan, Laura C Bell
@email: sudarshan.ragunathan@barrowneuro.org
@institution: Barrow Neurological Institute, Phoenix, AZ, USA
@lab: Translational Bioimaging Group (TBG)

DESCRIPTION
-----------
This function performs a deconvolution through two analysis steps.
The first step in performing deconvolution is the discretization of the AIF. (DOI:10.1088/0031-9155/52/22/014).
The second step is the regularization choice (DOI:10.1088/0031-9155/52/2/009).
Based on these articles, we have implemented a Volterra discretization with L-curve criterion
for regularization to demonstrate derivation of of the residue function needed for CBF and MTT calculations.

Input(s)/Ouput(s):
    INPUTS:
        dR2s_AIF - AIF ∆R2* values [s-1]
        dR2s_leakagecorrection - Leakage corrected tissue ∆R2* values [s-1]
        TR - Recovery Time [s]
    OUTPUTS:
        mu_opt - The optimal mu for regularization using an L-curve criterion method
        B -
        S -
"""
import numpy as np
from numpy.linalg import svd


def AIFdeconvolution(dR2s_AIF, dR2s_leakagecorrected, TR):
    # Discretize the AIF
    nt = np.shape(dR2s_leakagecorrected)[0]
    A_mtx = np.zeros([nt, nt])
    for i in range(nt):
        for j in range(nt):
            if j == 0 and i != 0:
                A_mtx[i, j] = (2 * dR2s_AIF[i] + dR2s_AIF[i - 1]) / 6.0
            elif i == j:
                A_mtx[i, j] = (2 * dR2s_AIF[0] + dR2s_AIF[1]) / 6.0
            elif 0 < j and j < i:
                A_mtx[i, j] = ((2 * dR2s_AIF[i - j] + dR2s_AIF[i - j - 1]) / 6) + (
                    (2 * dR2s_AIF[i - j] + dR2s_AIF[i - j + 1]) / 6
                )
        else:
            A_mtx[i, j] = 0.0

    # Singular value Decomposition (SVD) without regularization
    A_mtx = TR * A_mtx
    B0 = dR2s_leakagecorrected

    U, S, V = svd(A_mtx)
    S_d = np.diag(S)
    B = np.transpose(U) @ B0

    # Start L-curve regularization to get optimal mu used for regularization
    umax = 10.0
    umin = 10e-10
    nu = 400
    k = np.arange(nu)
    u = np.amax(S) * umin * np.power((umax / umin), ((k - 1) / (nu - 1)))

    l_0 = np.zeros([nu, A_mtx[:, 0].size])
    l_1 = np.zeros([nu, A_mtx[:, 0].size])
    l_2 = np.zeros([nu, A_mtx[:, 0].size])
    L = np.zeros([nu, A_mtx[:, 0].size, 3])
    for x in range(nu):
        for y in range(A_mtx[:, 0].size):
            l_0[x, y] = np.power(
                (np.power(u[x], 2) / (np.power(S[y], 2) + np.power(u[x], 2))), 2
            )
            l_1[x, y] = np.power((S[y] / (np.power(S[y], 2) + np.power(u[x], 2))), 2)
            l_2[x, y] = ((-4) * u[x] * np.power(S[y], 2)) / np.power(
                (np.power(S[y], 2) + np.power(u[x], 2)), 3
            )

    L[:, :, 0] = l_0
    L[:, :, 1] = l_1
    L[:, :, 2] = l_2

    # Start LCCOPTIMIZE
    k = (nu - 1) - 1
    m = np.zeros([nu, 3])
    product = np.zeros(A_mtx[:, 0].size)
    L_curve = np.zeros(nu)

    for x in range(A_mtx[:, 0].size):
        U_i = U[:, x]
        product[x] = np.power((np.transpose(U_i) @ B), 2)

    for x in range(3):
        l_tmp = L[:, :, x]
        m[:, x] = np.sum(l_tmp, axis=1) * np.sum(product)

    for x in range(nu):
        L_curve[x] = (
            2
            * (m[x, 1] * m[x, 0] / m[x, 2])
            * (
                (
                    np.power(u[x], 2) * m[x, 2] * m[x, 0]
                    + 2 * u[x] * m[x, 1] * m[x, 0]
                    + np.power(u[x], 4) * m[x, 1] * m[x, 2]
                )
                / np.power(
                    (np.power(u[x], 4) * np.power(m[x, 1], 2) + np.power(m[x, 0], 2)),
                    (3 / 2),
                )
            )
        )

    L_minus1 = L_curve[k - 2]
    L_0 = L_curve[k - 1]
    L_1 = L_curve[k]

    while L_0 >= L_minus1 or L_0 >= L_1:
        k = k - 1

        L_1 = L_0
        L_0 = L_minus1
        L_minus1 = L_curve[k - 1]
        mu_opt = u[k - 1]

    Bpi = np.multiply(B, np.divide(S, (np.power(S, 2) + np.power(mu_opt, 2))))
    Rf = np.transpose(V) @ Bpi

    return Rf, mu_opt, B, S
