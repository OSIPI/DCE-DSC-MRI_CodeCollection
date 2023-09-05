import sys, os
import MRImageAnalysis as mri
import matplotlib.pyplot as plt
import numpy as np


savePath = '../results/bat/parameter_estimates/'
K_transs = np.linspace(0.04, 0.14, 50)
v_p      = 0.02
v_e      = 0.2
F_ps     = [0.2, 0.4, 0.8]



def f(t, i):
	'''
	i is how many dts to move right
	'''
	ret = np.zeros(len(t))
	ret[i] = 1
	return ret


t0, C_a0 = mri.DCE.AIF.loadStandard()
dt = t0[1]-t0[0]

# remove the baseline in the AIF
c = 0
for i in (C_a0>0)*1:
	if i == 1:
		first = c
		break
	c += 1
original_BAT = t0[c]
C_a = np.zeros(len(C_a0))
C_a[:-c] = C_a0[c:]
C_a[-c:] = C_a0[-1]


BATs = np.arange(original_BAT-4, original_BAT+4, 0.1)

C_a_new = np.zeros((len(BATs), len(C_a)))
for i in range(len(BATs)):
	C_a_new[i] = np.convolve(C_a, f(t0, int(BATs[i]/dt)))[:len(t0)]


Ktrans_values = np.zeros((len(BATs), len(K_transs), len(F_ps)))
ve_values     = np.zeros((len(BATs), len(K_transs), len(F_ps)))
vp_values     = np.zeros((len(BATs), len(K_transs), len(F_ps)))
Fp_values     = np.zeros((len(BATs), len(K_transs), len(F_ps)))
dBAT = BATs - original_BAT

t0 /= 60 # time in minutes
pbar = mri.io.ProgressBar('', maxval=len(K_transs)*len(F_ps))
for i in range(len(K_transs)):
	for j in range(len(F_ps)):
		# create a signal with appropriate values
		S0 = mri.DCE.Models.twoCXM(t0, C_a0, K_trans=K_transs[i], v_p=v_p, v_e=v_e, F_p=F_ps[j])
		# now downsample the signal so that we get a better dt
		dt = 2/60.
		t, S = mri.math.misc.downSampleAverage(t0, S0, dt)

		# compute the model fit using different AIFs with different BATs
		for k in range(len(BATs)):
			_, C_a = mri.math.misc.downSampleAverage(t0, C_a_new[k], dt)
			fit = mri.DCE.Analyze.fitToModel('2CXM', S, t, C_a, showPbar=False)
			if fit.v_p < 0 or fit.v_e < 0 or fit.v_p > 100 or fit.v_e > 100 or fit.v_e+fit.v_p>100:
				_K_trans = np.inf
				_v_e     = np.inf
				_v_p     = np.inf
				_F_p     = np.inf
			elif fit.K_trans > fit.F_p or fit.K_trans < 0 or fit.F_p < 0:
				_K_trans = np.inf
				_v_e     = np.inf
				_v_p     = np.inf
				_F_p     = np.inf
			else:
				_K_trans = fit.K_trans
				_v_e     = fit.v_e
				_v_p     = fit.v_p
				_F_p     = fit.F_p

			Ktrans_values[k, i, j] = _K_trans
			ve_values[k, i, j]     = _v_e
			vp_values[k, i, j]     = _v_p
			Fp_values[k, i, j]     = _F_p
		pbar.update()
pbar.finish()

for p in ['K_trans', 'v_e', 'v_p', 'F_p']:
	for j in range(len(F_ps)):
		save = np.zeros((len(BATs), len(K_transs)))
		for i in range(len(K_transs)):
			if p == 'K_trans':
				save[:,i] = Ktrans_values[:,i,j]*100
			if p == 'v_e':
				save[:,i] = ve_values[:,i,j]*100
			if p == 'v_p':
				save[:,i] = vp_values[:,i,j]*100
			if p == 'F_p':
				save[:,i] = Fp_values[:,i,j]*100
		header = 'The rows are values at different changes in bolus arrival times (BAT) ranging from {} to {} with N={} values'.format(dBAT[0], dBAT[-1], len(dBAT))
		header += '\nThen the columns are estimated {} for different values of K_trans. The first column is for K_trans={}, the second for K_trans={},'.format(p, K_transs[0]*100,  K_transs[1]*100)
		header += '\nand so on untill K_trans={}. Total number of K_trans values is N={}.'.format(K_transs[-1]*100, len(K_transs))
		header += '\nTrue values: v_p={} ml/100g, v_e={} ml/100g.'.format(v_p*100, v_e*100)
		np.savetxt(savePath+'{}_estimate_Fp={}.txt'.format(p, int(F_ps[j]*100)), save, header=header)
sys.exit()










