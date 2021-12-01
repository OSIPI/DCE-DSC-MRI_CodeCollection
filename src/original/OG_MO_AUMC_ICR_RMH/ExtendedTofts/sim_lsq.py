"""
March 2021 by Oliver Gurney-Champion and Matthew Orton
o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion
Solves the Extended Tofts model for each voxel and returns model parameters using the computationally efficient AIF as described by Orton et al. 2008 in https://doi.org/10.1088/0031-9155/53/5/005
The Cosine8 AIF is further described by Rata et al. 2016 in https://doi.org/10.1007/s00330-015-4012-9

Copyright (C) 2021 by Oliver Gurney-Champion and Matthew Orton

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

requirements:
scipy
joblib
matplotlib
numpy
"""

import numpy as np
import DCE as dce
import time
import hpexample
import matplotlib.pyplot as plt

SNR=50
hp=hpexample.Hyperparams()
# example AIF function that is population based AIF for pancreatic cancer patients (plasma concentration): https://doi.org/10.1016/j.mri.2018.02.005 and used in https://doi.org/10.1002/1878-0261.12688
example_AIF = [0.4120, 0.1867, 0.0309, 0.0333, 0.0252,0.0391, 0.0257, 0.0278, 0.0356, 0.0336, 0.0450, 0.0428, 0.0405, 0.0479, 0.0615, 0.0908, 0.1787, 0.5667, 1.9445, 4.6015, 6.5225, 8.7509, 6.9091,    6.0901,    4.6583,    3.5056,    2.8350,    2.3160,    1.9862,   1.8966,    1.8934,    2.0551,    2.1454,    2.1860,    2.1717,    2.1783,    2.0682,    1.9374,    1.8774,    1.8100,    1.7271,    1.6808,    1.6205,    1.6182,    1.5581,    1.5831,    1.5469,    1.5513,    1.5469,    1.5393,    1.5398,    1.5243,    1.5588,    1.5137,    1.4882,    1.5302,    1.4465,    1.4588,    1.4582,    1.4741,    1.4251,    1.4850,    1.4249,    1.4075,    1.4414,    1.4190,    1.3938,    1.4047,    1.3961,    1.3857,    1.3794,    1.3691,    1.3585,    1.3784,    1.3299,    1.3100,    1.3231,    1.3130,    1.3053,    1.3129,    1.2640,    1.3032,    1.2721,    1.2649,    1.2426,    1.2345,    1.2416,    1.2340,    1.2172,    1.2128,    1.2374,    1.2213,    1.2012,    1.1947,    1.1851,    1.1543,    1.1841,    1.1892,    1.1636,    1.1522,    1.2028,    1.1482,    1.1751,    1.1605,    1.1531,    1.1628,    1.1232,    1.1255,    1.1227,    1.1362,    1.1002,    1.1486,    1.1178,    1.0933,    1.1085,    1.0972,    1.0982,    1.0860,    1.1026,    1.0679,    1.0670,    1.0870,    1.0640,    1.0839,    1.0671,    1.0452,    1.0639,    1.0490,    1.0358,    1.0360,    1.0378,    1.0258,    1.0371,    1.0287,    1.0300,    1.0171,    1.0195,    1.0044,    1.0052,    1.0154,    1.0510,    1.0066,    1.0161,    0.9986,    0.9653,    0.9922,    0.9914,    0.9749,    0.9934,    0.9621,    0.9733,    0.9894,    0.9900,    0.9754,    0.9240,    0.9347,    0.8556,    0.7921,    0.7380,    0.6348]

#rep1 is pre-bolus low FA measurements
rep1 = hp.acquisition.rep1
#rep2 is number of measurements during contrast injection
rep2 = len(example_AIF)

timing = np.array(range(0, rep2)) * hp.acquisition.time / 60

# two population-based AIFs (plasma concentration)
aifHN = dce.aifPopHN()
aifPMB = dce.aifPopPMB()

# choose model, either 'Cosine8' or 'Cosine4'
model = 'Cosine8'
aif = dce.fit_aif(example_AIF, timing, model=model)

plt.plot(timing, dce.Cosine4AIF(timing,aifHN['ab'],aifHN['ae'],aifHN['mb'],aifHN['me'],aifHN['t0']), label='Population AIF from H&N study JNM') #https://doi.org/10.2967/jnumed.116.174433
plt.plot(timing, dce.Cosine4AIF(timing,aifPMB['ab'],aifPMB['ae'],aifPMB['mb'],aifPMB['me'],aifPMB['t0']), label='Population AIF from PMB/MRM papers') #https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21066 and https://iopscience.iop.org/article/10.1088/0031-9155/53/5/005
plt.plot(timing, dce.Cosine8AIF(timing, aif['ab'], aif['ar'], aif['ae'], aif['mb'], aif['mm'], aif['mr'], aif['me'], aif['tr'], aif['t0']), label='Model fit to the measured AIF in the abdomen (MRM/Mol Onc.)') #https://doi.org/10.1016/j.mri.2018.02.005 and used in https://doi.org/10.1002/1878-0261.12688
plt.plot(timing, example_AIF, marker='.', markersize=3, linestyle='', label='Measured AIF from the abdomen')
plt.legend()
plt.show()

timing = np.array(range(0, rep2)) * hp.acquisition.time / 60
aif = dce.fit_aif(example_AIF, timing, model=hp.model)

# correct our population-based AIF for patient-specific Hct
Hct=0.4
aif['ab']=aif['ab']/ (1. - Hct)

# we will simulate a matrix with signals over time (y) from different voxels (x) and then fit these and determine the errors in the fits.

# there are two sets of Flip Angles during these reps

timing = np.array(range(0, rep2 + rep1)) * hp.acquisition.time / 60
state = 123
# simulate distributions of data
rg = np.random.RandomState(state)
test = rg.uniform(0, 1, (hp.simulations.num_samples, 1))
vp = hp.simulations.vp_min + (test * (hp.simulations.vp_max - hp.simulations.vp_min))
test = rg.uniform(0, 1, (hp.simulations.num_samples, 1))
ve = hp.simulations.ve_min + (test * (hp.simulations.ve_max - hp.simulations.ve_min))
test = rg.uniform(0, 1, (hp.simulations.num_samples, 1))
kep = hp.simulations.kep_min + (test * (hp.simulations.kep_max - hp.simulations.kep_min))
Tonset = rg.uniform(hp.acquisition.Tonset_min, hp.acquisition.Tonset_max, (hp.simulations.num_samples, 1))
test = rg.uniform(0, 1, (hp.simulations.num_samples, 1))
R1 = hp.simulations.R1_min + (test * (hp.simulations.R1_max - hp.simulations.R1_min))
# here we put both FA's during the right time periods
hp.acquisition.FAlist = np.append(np.tile(hp.acquisition.FA1, rep1), np.tile(hp.acquisition.FA2, rep2))

del test
## generate dw signals and store them in X_dw
# make time parameter
num_time = len(timing)
# make empty data matrixes
X_dw = np.zeros((hp.simulations.num_samples, num_time))
R1eff = np.zeros((hp.simulations.num_samples, num_time))
C = np.zeros((hp.simulations.num_samples, num_time))
test = rg.uniform(0, 1, (hp.simulations.num_samples))
Hct = None
## fill up the data
for aa in range(len(kep)):
    # first we calculate the concentration curve from the underlying parameters
    if hp.model=='Cosine4':
        C[aa, :] = dce.Cosine4AIF_ExtKety(timing, aif, kep[aa][0], (Tonset[aa][0] + rep1 * hp.acquisition.time) / 60,
                                              ve[aa][0], vp[aa][0])
    if hp.model=='Cosine8':
        C[aa, :] = dce.Cosine8AIF_ExtKety(timing, aif, kep[aa][0], (Tonset[aa][0] + rep1 * hp.acquisition.time) / 60,
                                              ve[aa][0], vp[aa][0])
    ## then we go from concentration curve to R1 (=1/T1) relaxation times
    R1eff[aa, :] = dce.con_to_R1eff(C[aa, :], R1[aa][0], hp.acquisition.r1)
    ## then we go from relaxation time to axtual image intensity
    X_dw[aa, :] = dce.r1eff_to_dce(R1eff[aa, :], hp.acquisition.TR, hp.acquisition.FAlist)
# normalise the signal
S0_out = np.mean(X_dw[:, rep1:rep1 + 10], axis=1)
dce_signal_scaled = X_dw / S0_out[:, None]
noise_imag = np.zeros([hp.simulations.num_samples, num_time])
noise_real = np.zeros([hp.simulations.num_samples, num_time])
#simulate noise to add
for i in range(0, hp.simulations.num_samples - 1):
    noise_real[i,] = rg.normal(0, 1/SNR,
                               (1, num_time))  # wrong! need a SD per input. Might need to loop to make noise
    noise_imag[i,] = rg.normal(0, 1/SNR, (1, num_time))
# add noise
dce_signal_scaled_noisy = np.sqrt(np.power(dce_signal_scaled + noise_real, 2) + np.power(noise_imag, 2))
S0_noisy = np.mean(dce_signal_scaled_noisy[:, rep1:rep1 + 10], axis=1)
dce_signal_noisy = dce_signal_scaled_noisy / S0_noisy[:, None]
del dce_signal_scaled_noisy, S0_noisy, noise_imag, noise_real, S0_out, X_dw, R1eff, C
## HEre  we now have simulated DCE data with double flip angle pre-contrast bolus

flip_angles = [hp.acquisition.FA1, hp.acquisition.FA2]
sigT1 = np.transpose([np.sqrt(np.mean(np.square(dce_signal_noisy[:, 0:rep1]), 1)),
                      np.sqrt(np.mean(np.square(dce_signal_noisy[:, rep1:rep1 + 10]), 1))])

#from the two FA data we just simulated, we now try to estimate R1 (=1/T1) directly. Note that during this time no contrast agent is injected jet
R1map = dce.R1_two_fas(sigT1, flip_angles, hp.acquisition.TR)
S0 = np.mean(dce_signal_noisy[:, rep1:rep1 + 10], axis=1)
# with baseline R1 and S0 we can estimate R1 over time (effected by the contrast agent arriving)
R1eff2 = dce.dce_to_r1eff(dce_signal_noisy[:, rep1:],  R1map, hp.acquisition.TR, hp.acquisition.FA2, 9)

C1 = dce.r1eff_to_conc(R1eff2, np.transpose([R1map]), hp.acquisition.r1)
del sigT1, R1map, S0, R1eff2
timing = np.array(range(0, rep2)) * hp.acquisition.time / 60
print(np.nanmean(C1))
start_time = time.time()
print('starting lsq fit')
# do the fit:
paramsf = dce.fit_tofts_model(C1, timing, aif, jobs=hp.jobs, model=hp.model)
elapsed_time = time.time() - start_time
print('time elapsed for lsqfit: {}'.format(elapsed_time))

import matplotlib.pyplot as plt
plt.plot(kep, np.expand_dims(paramsf[0],1), marker='.', linestyle='')
plt.show()

error_ke = np.expand_dims(paramsf[0],1) - kep
error_ve = np.expand_dims(paramsf[2],1) - ve
error_vp = np.expand_dims(paramsf[3],1) - vp
error_dt = np.expand_dims(paramsf[1],1) - Tonset / 60

randerror_ke = np.std(error_ke)
randerror_ve = np.std(error_ve)
randerror_vp = np.std(error_vp)
randerror_dt = np.std(error_dt)

syserror_ke = np.mean(error_ke)
syserror_ve = np.mean(error_ve)
syserror_vp = np.mean(error_vp)
syserror_dt = np.mean(error_dt)

del error_ke, error_ve, error_vp, error_dt, paramsf
normke = np.mean(kep)
normve = np.mean(ve)
normvp = np.mean(vp)
normdt = np.mean(Tonset / 60)

del kep, ve, vp, Tonset

print('ke_sim, dke_lsq, sys_ke_lsq, dke, sys_ke')
print([normke, '  ', randerror_ke, '  ', syserror_ke])
print([normve, '  ', randerror_ve, '  ', syserror_ve])
print([normvp, '  ', randerror_vp, '  ', syserror_vp])
print([normdt, '  ', randerror_dt, '  ', syserror_dt])