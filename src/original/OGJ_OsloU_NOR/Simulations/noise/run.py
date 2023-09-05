import MRImageAnalysis as mri
import numpy as np
import matplotlib.pyplot as plt

from phantoms_py.phantom import phantom
import sys

data_save_path = '../results/noise/'


def plot_phantom_curves(ph, fit=None):
	if fit is not None:
		fitted_img = fit.fittedImage()
	else:
		fitted_img = None

	fig, axs = plt.subplots(ph.noRows, ph.noCols, figsize=(15,8))
	for i in range(ph.noRows):
		for j in range(ph.noCols):
			starti = i*ph.sectionWidth+i*ph.sectionSpacing
			stopi  = (i+1)*ph.sectionWidth+i*ph.sectionSpacing

			startj = j*ph.sectionHeight+j*ph.sectionSpacing
			stopj  = (j+1)*ph.sectionHeight+j*ph.sectionSpacing

		
			axs[i,j].plot(ph.time, np.mean(ph[0,:,starti:stopi,startj:stopj], axis=(1,2)), 'k')
			if fit is not None:
				axs[i,j].plot(ph.time, np.mean(fitted_img[0,:,starti:stopi,startj:stopj], axis=(1,2)), 'r')
				
	for a in axs.flatten():
		a.tick_params(labelbottom='off',bottom='off',top='off',left='off',right='off',labelleft='off')

def add_noise(ph, SNR):	
	pbar = mri.io.ProgressBar('Adding noise...', maxval=ph[0,0].shape[0]*ph[0,0].shape[1]*len(ph[0,:,0,0]))
	for i in range(ph[0,0].shape[0]):
		for j in range(ph[0,0].shape[1]):
			for t in range(len(ph[0,:,0,0])):
				S = ph[0,t,i,j]
				ph[0,t,i,j] += np.random.normal(0, S/SNR[j], 1)
				pbar.update()

	pbar.finish()

	return ph

def down_sample(ph, dt):
	new_t, new_C_a = mri.math.misc.downSampleAverage(ph.time, ph.C_a, dt)
	new_voxelArray = np.zeros((1, len(new_t)+1, ph.shape[2], ph.shape[3]))
	pbar = mri.io.ProgressBar('Downsampling...', maxval=ph.shape[2]*ph.shape[3])
	for i in range(ph.shape[2]):
		for j in range(ph.shape[3]):
			new_t, new_S = mri.math.misc.downSampleAverage(ph.time, ph[0,:,i,j], dt)
			new_S = np.append(np.zeros(1), new_S)
			plt.show()
			new_voxelArray[0,:,i,j] = new_S

			pbar.update()
	pbar.finish()
	
	new_t = np.append(new_t, np.array([new_t[-1] + new_t[1] - new_t[0]]))
	
	ph.voxelArray = new_voxelArray
	ph.time = new_t
	ph.noTimepoints = len(new_t)
	ph.C_a = np.append(np.zeros(1), new_C_a)
	return ph


ph, data, _ = phantom('sim_with_noise')

noise = np.linspace(100,10,ph.noCols)
noise[0] = np.inf

ph = down_sample(ph, 1/60.)
ph = add_noise(ph, noise)


header = 'This file contains voxel by voxel analysis of a phantom.'
header += '\nIn each row, the value of K_trans is held constant, and the noise is changed in each column.'
header += '\nK_trans is varied along the rows. The gaussian noise is changed from SNR={} to SNR={},'.format(noise[0], noise[-1])
header += '\nand K_trans is varied from K_trans={} to K_trans={}'.format(data['K_trans'][0,0], data['K_trans'][-1,-1])

for model in ['2CXM', 'ETM', 'TM']:
	fit = mri.DCE.Analyze.fitToModel(model, ph.voxelArray, ph.time, ph.C_a)

	np.savetxt(data_save_path+model+'/K_trans.txt', fit.K_trans.voxelArray[0,0,:,:])
	np.savetxt(data_save_path+model+'/v_e.txt', fit.v_e.voxelArray[0,0,:,:])

	if model in ['ETM', '2CXM']:
		np.savetxt(data_save_path+model+'/v_p.txt', fit.v_p.voxelArray[0,0,:,:])
	if model == '2CXM':
		np.savetxt(data_save_path+model+'/F_p.txt', fit.F_p.voxelArray[0,0,:,:])



























