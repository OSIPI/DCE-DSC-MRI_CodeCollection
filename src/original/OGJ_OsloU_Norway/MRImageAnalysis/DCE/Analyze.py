import numpy as np
import scipy.optimize
from .. import math
from .. import io
from .. import MRImage
from . import Models
import multiprocessing as mp

def fitToModel(model, img, t, C_a, integrationMethod='trapezoidal', method='LLSQ', showPbar=True):
	if model == 'TM':
		return ToftsModel(img, t, C_a, integrationMethod=integrationMethod, method=method, showPbar=showPbar)
	if model == 'ETM':
		return ExtendedToftsModel(img, t, C_a, integrationMethod=integrationMethod, method=method, showPbar=showPbar)
	if model == '2CXM' or model == 'twoCXM' or model == 'TCx':
		return TwoCompartmentExchangeModel(img, t, C_a, integrationMethod=integrationMethod, method=method, showPbar=showPbar)

class ToftsModel:
	def __init__(self, img, t, C_a, integrationMethod='trapezoidal', method='LLSQ', showPbar=True):
		self.t = t
		self.C_a = C_a

		self.isSingleSignal = False
		if len(img.shape) == 1:
			I = np.zeros((1,len(img),1,1))
			I[0,:,0,0] = img
			img = I
			self.isSingleSignal = True

		if method == 'LLSQ':
			self.modelFit, self.Rsq = _fitToModelLLSQ('TM', img, t, C_a, integrationMethod, showPbar)
		if method == 'NLLS':
			self.modelFit = _fitToModelNLLS('TM', img, t, C_a)
		if method == 'NLLSp':
			self.modelFit = _fitToModelNLLS_paralell('TM', img, t, C_a)

	@property
	def K_trans(self):
		if self.isSingleSignal:
			return self.modelFit[:,0:1,:,:][0,0,0,0]
		return MRImage.MRImage(self.modelFit[:,0:1,:,:])

	@property
	def k_ep(self):
		if self.isSingleSignal:
			return self.modelFit[:,1:2,:,:][0,0,0,0]
		return MRImage.MRImage(self.modelFit[:,1:2,:,:])

	@property
	def v_e(self):
		if self.isSingleSignal:
			return self.K_trans/self.k_ep
		return MRImage.MRImage(self.modelFit[:,0:1,:,:]/self.modelFit[:,1:2,:,:])

	def fittedCurve(self, i=0, j=0, slc=0):
		if self.isSingleSignal:
			return Models.TM(self.t, self.C_a, K_trans=self.K_trans, k_ep=self.k_ep)
		return Models.TM(self.t, self.C_a, K_trans=self.K_trans[slc,0,i,j], k_ep=self.k_ep[slc,0,i,j])
	
	def fittedImage(self):
		if hasattr(self, 'fittedImg'):
			return self.fittedImg
		fittedImg = np.zeros((self.modelFit.shape[0], len(self.t), self.modelFit.shape[2], self.modelFit.shape[3]))
		for slc in range(fittedImg.shape[0]):
			for i in range(fittedImg.shape[2]):
				for j in range(fittedImg.shape[3]):
					fittedImg[slc,:,i,j] = Models.TM(self.t, self.C_a, K_trans=self.K_trans[slc,0,i,j], k_ep=self.k_ep[slc,0,i,j])

		self.fittedImg = MRImage.MRImage(fittedImg)
		return self.fittedImg

class ExtendedToftsModel:
	def __init__(self, img, t, C_a, integrationMethod='trapezoidal', method='LLSQ', showPbar=True):
		self.t = t
		self.C_a = C_a

		self.isSingleSignal = False
		if len(img.shape) == 1:
			I = np.zeros((1,len(img),1,1))
			I[0,:,0,0] = img
			img = I
			self.isSingleSignal = True

		if method == 'LLSQ':
			self.modelFit, self.Rsq = _fitToModelLLSQ('ETM', img, t, C_a, integrationMethod, showPbar)
		if method == 'NLLS':
			self.modelFit = _fitToModelNLLS('ETM', img, t, C_a)
		if method == 'NLLSp':
			self.modelFit = _fitToModelNLLS_paralell('ETM', img, t, C_a)

	@property
	def K_trans(self):
		if self.isSingleSignal:
			return self.modelFit[:,0:1,:,:][0,0,0,0]
		return MRImage.MRImage(self.modelFit[:,0:1,:,:])

	@property
	def k_ep(self):
		if self.isSingleSignal:
			return self.modelFit[:,1:2,:,:][0,0,0,0]
		return MRImage.MRImage(self.modelFit[:,1:2,:,:])

	@property
	def v_p(self):
		if self.isSingleSignal:
			return self.modelFit[:,2:3,:,:][0,0,0,0]
		return MRImage.MRImage(self.modelFit[:,2:3,:,:])

	@property
	def v_e(self):
		if self.isSingleSignal:
			return self.modelFit[:,3:4,:,:][0,0,0,0]
		return MRImage.MRImage(self.modelFit[:,3:4,:,:])

	def fittedCurve(self, i=0, j=0, slc=0):
		if self.isSingleSignal:
			return Models.ETM(self.t, self.C_a, K_trans=self.K_trans, k_ep=self.k_ep, v_p=self.v_p)
		return Models.ETM(self.t, self.C_a, K_trans=self.K_trans[slc,0,i,j], k_ep=self.k_ep[slc,0,i,j], v_p=self.v_p[slc,0,i,j])

	def fittedImage(self):
		if hasattr(self, 'fittedImg'):
			return self.fittedImg
		fittedImg = np.zeros((self.modelFit.shape[0], len(self.t), self.modelFit.shape[2], self.modelFit.shape[3]))
		for slc in range(fittedImg.shape[0]):
			for i in range(fittedImg.shape[2]):
				for j in range(fittedImg.shape[3]):
					fittedImg[slc,:,i,j] = Models.ETM(self.t, self.C_a, K_trans=self.K_trans[slc,0,i,j], k_ep=self.k_ep[slc,0,i,j], v_p=self.v_p[slc,0,i,j])

		self.fittedImg = MRImage.MRImage(fittedImg)
		return self.fittedImg

class TwoCompartmentExchangeModel:
	def __init__(self, img, t, C_a, integrationMethod='trapezoidal', method='LLSQ', showPbar=True, weights=None):
		self.t = t
		self.C_a = C_a

		self.isSingleSignal = False
		if len(img.shape) == 1:
			I = np.zeros((1,len(img),1,1))
			I[0,:,0,0] = img
			img = I
			self.isSingleSignal = True

		if method == 'LLSQ':
			self.modelFit, self.Rsq = _fitToModelLLSQ('twoCXM', img, t, C_a, integrationMethod, showPbar, weights=weights)
		if method == 'NNLS':
			raise NotImplementedError('The non linear method is not yes implemented for 2CXM')
		if method == 'NNLSp':
			raise NotImplementedError('The non linear method (paralell) is not yet implemented for 2CXM')
	
	@property
	def PS(self):
		if self.isSingleSignal:
			return self.modelFit[:,0:1,:,:][0,0,0,0]
		return MRImage.MRImage(self.modelFit[:,0:1,:,:])

	@property
	def F_p(self):
		if self.isSingleSignal:
			return self.modelFit[:,1:2,:,:][0,0,0,0]
		return MRImage.MRImage(self.modelFit[:,1:2,:,:])

	@property
	def v_p(self):
		if self.isSingleSignal:
			return self.modelFit[:,2:3,:,:][0,0,0,0]
		return MRImage.MRImage(self.modelFit[:,2:3,:,:])

	@property
	def v_e(self):
		if self.isSingleSignal:
			return self.modelFit[:,3:4,:,:][0,0,0,0]
		return MRImage.MRImage(self.modelFit[:,3:4,:,:])

	@property
	def K_trans(self):
		if self.isSingleSignal:
			return Models.Conversion.K_trans(self.PS, self.F_p)
		return MRImage.MRImage(Models.Conversion.K_trans(self.PS.voxelArray, self.F_p.voxelArray))

	@property
	def k_ep(self):
		if self.isSingleSignal:
			return Models.Conversion.k_ep(K_trans=self.K_trans, v_e=self.v_e)
		return MRImage.MRImage(Models.Conversion.k_ep(K_trans=self.K_trans.voxelArray, v_e=self.v_e.voxelArray))

	def fittedCurve(self, i=0, j=0, slc=0):
		if self.isSingleSignal:
			return Models.twoCXM(self.t, self.C_a, PS=self.PS, F_p=self.F_p, v_e=self.v_e, v_p=self.v_p)
		return Models.twoCXM(self.t, self.C_a, PS=self.PS[slc,0,i,j], F_p=self.F_p[slc,0,i,j], v_e=self.v_e[slc,0,i,j], v_p=self.v_p[slc,0,i,j])

	def fittedImage(self):
		if hasattr(self, 'fittedImg'):
			return self.fittedImg
		fittedImg = np.zeros((self.modelFit.shape[0], len(self.t), self.modelFit.shape[2], self.modelFit.shape[3]))
		for slc in range(fittedImg.shape[0]):
			for i in range(fittedImg.shape[2]):
				for j in range(fittedImg.shape[3]):
					fittedImg[slc,:,i,j] = Models.twoCXM(self.t, self.C_a, PS=self.PS[slc,0,i,j], F_p=self.F_p[slc,0,i,j], v_e=self.v_e[slc,0,i,j], v_p=self.v_p[slc,0,i,j])

		self.fittedImg = MRImage.MRImage(fittedImg)
		return self.fittedImg

def _fitToModelLLSQ(model, img, t, C_a, integrationMethod, showPbar, weights=None):
	'''
		Function to calculate perfusion paramaters of an image using
		the linear least squares method.

		args:
			model:
				string of model to be used. Allowed values are 'TM' and 'ETM' for
				'Tofts model' and 'Extended Tofts model' respectively
			img:
				A 4D numpy array containing at least one time signal
			C_a:
				The arterial input function to be used
			integrationMethod:
				which integration method should be used.

		returns:
			array containing the parameters of each time series (pixel) in the original image.

			if model is TM then
				parameterImage[slc,0,i,j] = K_trans
				parameterImage[slc,1,i,j] = k_ep

			if model is ETM then
				parameterImage[slc,0,i,j] = K_trans
				parameterImage[slc,1,i,j] = k_ep
				parameterImage[slc,2,i,j] = v_p
				parameterImage[slc,3,i,j] = v_e

			if model is 2CXM then
				parameterImage[slc,0,i,j] = PS 
				parameterImage[slc,1,i,j] = F_p 
				parameterImage[slc,2,i,j] = v_p 
				parameterImage[slc,3,i,j] = v_e 
	'''

	A = eval('_initiateLstsqMatrix_'+model)(t, C_a, integrationMethod, weights=weights)

	nrows   = img[0][0].shape[0]
	ncols   = img[0][0].shape[1]
	nslices = len(img)

	# initiate an array to store the paramaters
	if model == 'TM':
		parameterImage = np.zeros((nslices,2,nrows,ncols))
	elif model in ['ETM', 'twoCXM']:
		parameterImage = np.zeros((nslices,4,nrows,ncols))
	Rsquared = np.zeros((nslices,nrows,ncols))

	pbar = io.ProgressBar('Calculating slice 1/{}'.format(len(img)), len(img)*nrows*ncols, show=showPbar)
	for slc in range(nslices):
		for i in range(nrows):
			for j in range(ncols):
				# get the signal for the current pixel
				signal = img[slc,:,i,j]

				# update the least squares matrix
				A = eval('_updateLstsqMatrix_'+model)(A, t, signal, integrationMethod, weights=weights)

				if model == 'TM':
					# calculate the model parameters
					B = np.linalg.lstsq(A, signal)
					K_trans = B[0][0] # K_trans
					k_ep    = B[0][1] # k_ep
					if K_trans < 0 or k_ep < 0:
						K_trans = np.inf
						k_ep    = np.inf
					parameterImage[slc,0,i,j] = K_trans
					parameterImage[slc,1,i,j] = k_ep
				if model == 'ETM':
					# calculate the model parameters
					B = np.linalg.lstsq(A, signal)

					K_trans = B[0][0] - B[0][1]*B[0][2] # K_trans
					k_ep    = B[0][1] # k_ep
					v_p     = B[0][2] # v_p
					v_e     = K_trans/B[0][1] # v_e
					if K_trans < 0 or k_ep < 0 or v_e < 0 or v_p < 0 or v_e > 1 or v_p > 1:
						K_trans = np.inf
						k_ep    = np.inf
						v_e     = np.inf
						v_p     = np.inf
					parameterImage[slc,0,i,j] = K_trans
					parameterImage[slc,1,i,j] = k_ep
					parameterImage[slc,2,i,j] = v_p
					parameterImage[slc,3,i,j] = v_e
				if model == 'twoCXM':
					# calculate the model parameters
					B = np.linalg.lstsq(A, signal)

					F_p = B[0][3]
					v_p = B[0][3]**2/(B[0][1]*B[0][3] - B[0][2])
					v_e = B[0][2]/B[0][0] - v_p
					PS  = B[0][0]*v_e*v_p/F_p
					K_trans = Models.Conversion.K_trans(PS=PS, F_p=F_p)
					if K_trans > F_p or F_p < 0 or K_trans < 0 or PS < 0 or v_e < 0 or v_p < 0 or v_e > 1 or v_p > 1 or v_e + v_p > 1:
						F_p = np.inf
						v_e = np.inf
						v_p = np.inf
						PS  = np.inf
					parameterImage[slc,0,i,j] = PS  # PS
					parameterImage[slc,1,i,j] = F_p # F_p
					parameterImage[slc,2,i,j] = v_p # v_p
					parameterImage[slc,3,i,j] = v_e # v_e
				# Rsquared[slc,i,j] = 1 - B[1]/(signal.size * signal.var())

				# update the progressbar
				pbar.update()
		pbar.update(-1, 'Calculating slice {}/{}'.format(slc+1,len(img)))
	pbar.finish()

	return MRImage.MRImage(parameterImage), Rsquared

def _initiateLstsqMatrix_TM(t, C_a, integrationMethod, weights=None):
	if weights is not None:
		io.printWarning('A weighting paradigm is not implemented for Tofts Model. The entered weights are ignored.')
	A = np.zeros((len(C_a), 2))
	A[:, 0] = math.Integration.cumulativeIntegral(t, C_a, integrationMethod)
	return A

def _updateLstsqMatrix_TM(A, t, C_t, integrationMethod, weights=None):
	if weights is not None:
		io.printWarning('A weighting paradigm is not implemented for Tofts Model. The entered weights are ignored.')
	A[:, 1] = -math.Integration.cumulativeIntegral(t, C_t, integrationMethod)
	return A

def _initiateLstsqMatrix_ETM(t, C_a, integrationMethod, weights=None):
	if weights is not None:
		io.printWarning('A weighting paradigm is not implemented for Extended Tofts Model. The entered weights are ignored.')
	A = np.zeros((len(C_a), 3))
	A[:, 0] = math.Integration.cumulativeIntegral(t, C_a, integrationMethod)
	A[:, 2] = C_a
	return A

def _updateLstsqMatrix_ETM(A, t, C_t, integrationMethod, weights=None):
	if weights is not None:
		io.printWarning('A weighting paradigm is not implemented for Extended Tofts Model. The entered weights are ignored.')
	return _updateLstsqMatrix_TM(A, t, C_t, integrationMethod)

def _initiateLstsqMatrix_twoCXM(t, C_a, integrationMethod, weights=None):
	if weights is None:
		weights = np.ones(len(t))
	A = np.zeros((len(C_a), 4))
	firstIntegral = math.Integration.cumulativeIntegral(t, C_a, integrationMethod)
	A[:, 3] = firstIntegral*weights
	A[:, 2] = math.Integration.cumulativeIntegral(t, firstIntegral, integrationMethod)*weights
	return A

def _updateLstsqMatrix_twoCXM(A, t, C_t, integrationMethod, weights=None):
	if weights is None:
		weights = np.ones(len(t))
	firstIntegral = math.Integration.cumulativeIntegral(t, C_t, integrationMethod)
	A[:, 1] = -firstIntegral*weights
	A[:, 0] = -math.Integration.cumulativeIntegral(t, firstIntegral, integrationMethod)*weights
	return A

def _fitToModelNLLS(model, img, t, C_a):
	nRows   = img[0][0].shape[0]
	nCols   = img[0][0].shape[1]
	nSlices = len(img)

	if model == 'TM':
		TM  = lambda t, K_trans, k_ep: Models.TM(t, C_a, K_trans, k_ep)
		parameterImage = np.zeros((nSlices, 2, nRows, nCols))
		pbar = io.ProgressBar('Calculating slice 1/{}'.format(nSlices), nSlices*nRows*nCols)
		for slc in range(nSlices):
			for i in range(nRows):
				for j in range(nCols):
					signal = img[slc,:,i,j]
					fit = scipy.optimize.curve_fit(TM, t, signal)

					parameterImage[slc,0,i,j] = fit[0][0] # K_trans
					parameterImage[slc,1,i,j] = fit[0][1] # k_ep
					pbar.update()
			pbar.update(-1, 'Calculating slice {}/{}'.format(slc+1,len(img)))
		pbar.finish()

	if model == 'ETM':
		ETM = lambda t, K_trans, k_ep, v_p: Models.ETM(t, C_a, K_trans, k_ep, v_p)
		parameterImage = np.zeros((nSlices, 4, nRows, nCols))

		for slc in range(nSlices):
			for i in range(nRows):
				for j in range(nCols):
					signal = img[slc, :, i, j]
					fit = scipy.optimize.curve_fit(ETM, t, signal, bounds=(0., [np.inf, np.inf, 1.]))

					parameterImage[slc,0,i,j] = fit[0][0] # K_trans
					parameterImage[slc,1,i,j] = fit[0][1] # k_ep
					parameterImage[slc,2,i,j] = fit[0][2] # v_p
					parameterImage[slc,3,i,j] = fit[0][0]/fit[0][1] # v_e = K_trans/k_ep
	
	return MRImage.MRImage(parameterImage)

def fit_signal(model, start, end, nCols, img, t, C_a, f, output):
	if model == 'TM':
		result = np.zeros((2, end-start, nCols))
		for i in range(len(result[0])):
			for j in range(len(result[0,0])):
				signal = img[:,start+i,j]
				fit = scipy.optimize.curve_fit(f, t, signal)
				result[0,i,j] = fit[0][0]
				result[1,i,j] = fit[0][1]
		output.put([start, end, result])
	if model == 'ETM':
		result = np.zeros((4, end-start, nCols))
		for i in range(len(result[0])):
			for j in range(len(result[0,0])):
				signal = img[:,start+i,j]
				fit = scipy.optimize.curve_fit(f, t, signal, bounds=(0., [np.inf, np.inf, 1.]))
				result[0,i,j] = fit[0][0] # K_trans
				result[1,i,j] = fit[0][1] # k_ep
				result[2,i,j] = fit[0][2] # v_p
				result[3,i,j] = fit[0][0]/fit[0][1] #v_e = K_trans/k_ep
		output.put([start, end, result])

def _fitToModelNLLS_paralell(model, img, t, C_a):
	'''
		function to calculate the model fit using
		paralell processing.
	'''
	nRows   = img[0][0].shape[0]
	nCols   = img[0][0].shape[1]
	nSlices = len(img)

	if nSlices > 1:
		print('Parallell processing only tested for single slice image. Use at own risk.')
		input('To continue, press the return key.')

	# compute how many rows to assign each core
	nprocs = mp.cpu_count()
	count = np.ones(nprocs)
	current_number = np.ceil(nRows/nprocs)
	i = 0
	while np.sum(count) < nRows:
		count[i] = current_number
		if np.sum(count) > nRows:
			count[i] = 1
			current_number -= 1
		else:
			i += 1
	count = np.insert(count, 0, 0)
	count = np.cumsum(count).astype(np.int)

	if model == 'TM':
		TM  = lambda t, K_trans, k_ep: Models.TM(t, C_a, K_trans, k_ep)
		parameterImage = np.zeros((nSlices, 2, nRows, nCols))
		for slc in range(nSlices):
			output = mp.Queue()
			processes = [mp.Process(target=fit_signal, args=('TM', count[i], count[i+1], nCols, img[0], t, C_a, TM, output)) for i in range(nprocs)]
			for p in processes:
				p.start()
			results = [output.get() for p in processes]
			for r in results:
				parameterImage[slc,0,r[0]:r[1],:] = r[2][0] # K_trans
				parameterImage[slc,1,r[0]:r[1],:] = r[2][1] # k_ep
	
	if model == 'ETM':
		ETM = lambda t, K_trans, k_ep, v_p: Models.ETM(t, C_a, K_trans, k_ep, v_p)
		parameterImage = np.zeros((nSlices, 4, nRows, nCols))
		for slc in range(nSlices):
			output = mp.Queue()
			processes = [mp.Process(target=fit_signal, args=('ETM', count[i], count[i+1], nCols, img[0], t, C_a, ETM, output)) for i in range(nprocs)]
			for p in processes:
				p.start()
			results = [output.get() for p in processes]
			for r in results:
				parameterImage[slc,0,r[0]:r[1],:] = r[2][0] # K_trans
				parameterImage[slc,1,r[0]:r[1],:] = r[2][1] # k_ep
				parameterImage[slc,2,r[0]:r[1],:] = r[2][2] # v_p
				parameterImage[slc,3,r[0]:r[1],:] = r[2][3] # v_e

	return MRImage.MRImage(parameterImage)