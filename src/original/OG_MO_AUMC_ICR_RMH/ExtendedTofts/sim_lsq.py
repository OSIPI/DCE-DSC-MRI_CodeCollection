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
# example AIF function that is population based AIF for pancreatic cancer patients (total signal): https://doi.org/10.1016/j.mri.2018.02.005 and used in https://doi.org/10.1002/1878-0261.12688
example_AIF = [0.2307,0.1046,0.0173,0.0186,0.0141,0.0219,0.0144,0.0156,0.0199,0.0188,0.0252,0.0239,0.0227,0.0268,0.0344,0.0509,0.1001,0.3174,1.0889,2.5769,3.6526,4.9005,3.8691,3.4104,2.6086,1.9631,1.5876,1.2970,1.1123,1.0621,1.0603,1.1509,1.2014,1.2242,1.2162,1.2198,1.1582,1.0849,1.0513,1.0136,0.9672,0.9413,0.9075,0.9062,0.8725,0.8866,0.8663,0.8687,0.8663,0.8620,0.8623,0.8536,0.8729,0.8477,0.8334,0.8569,0.8100,0.8169,0.8166,0.8255,0.7980,0.8316,0.7979,0.7882,0.8072,0.7946,0.7805,0.7867,0.7818,0.7760,0.7725,0.7667,0.7608,0.7719,0.7447,0.7336,0.7409,0.7353,0.7310,0.7352,0.7079,0.7298,0.7124,0.7083,0.6959,0.6913,0.6953,0.6910,0.6817,0.6792,0.6929,0.6839,0.6727,0.6690,0.6637,0.6464,0.6631,0.6659,0.6516,0.6452,0.6736,0.6430,0.6580,0.6499,0.6457,0.6512,0.6290,0.6303,0.6287,0.6363,0.6161,0.6432,0.6260,0.6122,0.6208,0.6144,0.6150,0.6082,0.6174,0.5980,0.5975,0.6087,0.5959,0.6070,0.5976,0.5853,0.5958,0.5874,0.5800,0.5801,0.5812,0.5744,0.5808,0.5760,0.5768,0.5696,0.5709,0.5624,0.5629,0.5686,0.5885,0.5637,0.5690,0.5592,0.5406,0.5556,0.5552,0.5460,0.5563,0.5388,0.5450,0.5541,0.5544,0.5462, 0.5174, 0.5235, 0.4792, 0.4436, 0.4133, 0.3555]

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