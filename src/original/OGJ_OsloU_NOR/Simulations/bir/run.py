import MRImageAnalysis as mri
import matplotlib.pyplot as plt
import numpy as np
import sys
from imports import *

savePath = "../results/bir/"
K_trans = 0.07
F_p = 0.40
v_e = 0.20
v_p = 0.02


BTTs = ["13.2", "20", "49"]

t0, _ = mri.DCE.AIF.loadStandard()
t0 /= 60  # convert to minutes
AIFs = np.zeros((len(t0), len(BTTs) + 1))
AIFs[:, 0] = t0 * 60

params = ["K_trans", "F_p", "v_p", "v_e"]

dts = np.arange(0.1, 25.1, 0.1) / 60


TM_values = np.zeros((len(dts), len(BTTs) * 2 + 1))
ETM_values = np.zeros((len(dts), len(BTTs) * 3 + 1))
twoCXM_values = np.zeros((len(dts), len(BTTs) * 4 + 1))

TM_values[:, 0] = dts * 60
ETM_values[:, 0] = dts * 60
twoCXM_values[:, 0] = dts * 60


for i in range(len(BTTs)):
    C_a0 = np.loadtxt("MRImageAnalysis/Data/AIF_BTT/{}.txt".format(BTTs[i]))
    AIFs[:, i + 1] = C_a0
    S0 = mri.DCE.Models.twoCXM(t0, C_a0, K_trans=K_trans, F_p=F_p, v_p=v_p, v_e=v_e)
    for j in range(len(dts)):
        t, C_a = mri.math.misc.downSampleAverage(t0, C_a0, dts[j])
        t, S = mri.math.misc.downSampleAverage(t0, S0, dts[j])

        C_a = np.append(np.array([0]), C_a)
        S = np.append(np.array([0]), S)
        t = np.append(t, np.array([t[-1] + t[1] - t[0]]))

        for model in ["TM", "ETM", "2CXM"]:
            fit = mri.DCE.Analyze.fitToModel(model, S, t, C_a, showPbar=False)
            if model == "TM":
                TM_values[j, i * 2 + 1] = fit.K_trans * 100
                TM_values[j, i * 2 + 2] = fit.v_e * 100
            if model == "ETM":
                ETM_values[j, i * 3 + 1] = fit.K_trans * 100
                ETM_values[j, i * 3 + 2] = fit.v_e * 100
                ETM_values[j, i * 3 + 3] = fit.v_p * 100
            if model == "2CXM":
                twoCXM_values[j, i * 4 + 1] = fit.K_trans * 100
                twoCXM_values[j, i * 4 + 2] = fit.v_e * 100
                twoCXM_values[j, i * 4 + 3] = fit.v_p * 100
                twoCXM_values[j, i * 4 + 4] = fit.F_p * 100


header = "True values are: K_trans = {}, v_e = {}, v_p = {}, F_p = {}".format(
    K_trans, v_e, v_p, F_p
)
header += "\nThe columns are:"
header += "\ndt (s)"
header_TM = header
header_ETM = header
header_2CXM = header
for BTT in BTTs:
    header_TM += ", K_trans(BTT = {}), v_e(BTT = {})".format(BTT, BTT)
    header_ETM += ", K_trans(BTT = {}), v_e(BTT = {}), v_p(BTT = {})".format(
        BTT, BTT, BTT
    )
    header_2CXM += (
        ", K_trans(BTT = {}), v_e(BTT = {}), v_p(BTT = {}), F_p(BTT = {})".format(
            BTT, BTT, BTT, BTT
        )
    )
np.savetxt(savePath + "TM_values.txt", TM_values, header=header_TM)
np.savetxt(savePath + "ETM_values.txt", ETM_values, header=header_ETM)
np.savetxt(savePath + "2CXM_values.txt", twoCXM_values, header=header_2CXM)
header = "t (s)"
for BTT in BTTs:
    header += ", AIF(BTT = {})".format(BTT)
np.savetxt(savePath + "AIFs.txt", AIFs, header=header)


"""

	Also do fft of the AIFs

"""

fft = None
fft1 = None
for i in range(len(BTTs)):
    frq, C_aFFT = dofft(t0 * 60, AIFs[:, i + 1])
    if fft is None:
        fft = np.zeros((len(frq), len(BTTs) + 1))
    fft[:, i + 1] = 20 * np.log10(abs(C_aFFT))

    frq1, C_aFFT1 = dofft(
        t0, twoCXM(t0, K_trans=0.07, F_p=[0.2, 0.4, 0.8][i], v_e=0.2, v_p=0.02)
    )
    if fft1 is None:
        fft1 = np.zeros((len(frq1), len(BTTs) + 1))
    fft1[:, i + 1] = 20 * np.log10(abs(C_aFFT1))

fft[:, 0] = frq
fft1[:, 0] = frq1 / 60


header = "FT of AIFs with different bolus transit times (BTT)."
header += "\nThe columns are with FT being in dB (20*log10(abs(FT))):"
header += "\nFrequency (Hz)"
for BTT in BTTs:
    header += ", FT(BTT={})".format(BTT)
np.savetxt(savePath + "AIF_FFT.txt", fft, header=header)

header = "FT of the 2CXM with different F_p (20, 40 and 80 ml/100g/min)."
header += "\nThe columns are with FT being in dB (20*log10(abs(FT))):"
header += "\nFrequency (Hz)"
for F_p in [20, 40, 80]:
    header += ", FT(F_p={})".format(F_p)
np.savetxt(savePath + "2CXM_FFT.txt", fft1, header=header)
