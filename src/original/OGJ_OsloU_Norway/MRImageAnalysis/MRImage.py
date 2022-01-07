import numpy as np
from . import io
from . import DCE
from . import math
import base64
#import cv2

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class MRImage:
	def __init__(self, voxelArray):
		self.voxelArray = voxelArray

		self.applyFilter = self.Filters(self)

	def __getitem__(self, key):
		return self.voxelArray[key]

	def __setitem__(self, key, value):
		self.voxelArray[key] = value

	def __bytes__(self):
		b64 = []
		for sl in self.imageArray:
			b64.append([])
			for img in sl:
				b64[-1].append(base64.encodestring(cv2.imencode('.jpg', img)[1]))
		return b64

	def __add__(self, other):
		if (type(other) == type(self)):
			return MRImage(self.voxelArray + other.voxelArray)
		else:
			return MRImage(self.voxelArray + other)

	def __sub__(self, other):
		if (type(other) == type(self)):
			return MRImage(self.voxelArray - other.voxelArray)
		else:
			return MRImage(self.voxelArray - other)

	def __mul__(self, other):
		if (type(other) == type(self)):
			return MRImage(self.voxelArray * other.voxelArray)
		else:
			return MRImage(self.voxelArray * other)

	def __truediv__(self, other):
		if (type(other) == type(self)):
			return MRImage(self.voxelArray / other.voxelArray)
		else:
			return MRImage(self.voxelArray / other)

	def __abs__(self):
		return MRImage(np.abs(self.voxelArray))

	def __str__(self):
		return str(self.voxelArray)

	@property
	def shape(self):
		return self.voxelArray.shape

	@property
	def nSlices(self):
		return self.voxelArray.shape[0]

	@property
	def nTimePoints(self):
		return self.voxelArray.shape[1]

	@property
	def nRows(self):
		return self.voxelArray.shape[2]

	@property
	def nColumns(self):
		return self.voxelArray.shape[3]
	@property
	def nCols(self):
		return self.nColumns

	def plot(self, **args):
		self.plotax = io.DisplayFunctions().imshow(self.voxelArray, **args)
		return self.plotax
	
	def colorbar(self):
		plt.colorbar(self.plotax.get_images()[0])

	def toNifti(self, filename, nImagesPerSlice=None, printData=False):
		if filename.split('.')[-1] not in ['nii', 'gz']:
			assert('File extension must be one of .nii and .nii.gz')
		if filename.split('.')[-1] == 'gz':
			if filename.split('.')[-2] != 'nii':
				assert('File extension must be one of .nii and .nii.gz')

		from medpy.io import save as save_as_nifti
		img     = self.voxelArray
		nSlices = img.shape[0]
		nTime   = img.shape[1]
		nRows   = img.shape[2]
		nCols   = img.shape[3]

		if nImagesPerSlice is not None:

			# extract only some timepoints
			time_idxs = np.arange(0, nTime, int(nTime/float(nImagesPerSlice)))
			
			t   = self.time.copy()
			C_a = self.C_a.copy()

			img = img[:,time_idxs,:,:]
			t   = t[time_idxs]
			C_a = C_a[time_idxs]
		else:
			time_idxs = np.arange(0, nTime, 1)
		new_image = np.zeros((nRows, nCols, nSlices, len(time_idxs)))

		for slc in range(nSlices):
			for ti in range(len(time_idxs)):
				new_image[:,:,slc,ti] = img[slc,ti,:,:]
		save_as_nifti(np.fliplr(np.rot90(new_image)), filename)

		if printData:
			print('Nifti image successfully saved to {}.'.format(filename))
			print('Temporal resolution: {} [unknown units]'.format(t[1]-t[0]))

	class Filters:
		def __init__(self, parent):
			self.parent = parent

		def simpleThreshold(self, lower=None, upper=None):
			if lower is not None:
				self.parent.voxelArray *= (self.parent.voxelArray > lower)*1
			if upper is not None:
				self.parent.voxelArray *= (self.parent.voxelArray < upper)*1

class MRRawImage(MRImage):
	def __init__(self, voxelArray, time, sliceLocations, FA, TR):
		MRImage.__init__(self, voxelArray)

		self.time           = time
		self.sliceLocations = sliceLocations
		self.FA             = FA
		self.TR             = TR

	def SI2R1(self, r_1=4.5, T_10=1.4, FA=None, TR=None):
		'''
			Converts an image, or a set of images, from signal
			intensity to relaxation rate.
			The values
				r1   = 4.5 s^-1 mM^-1
				T_10 = 1.4 s
			are commonly used values [1].

			We have
				R_1 = R_10 + r_1C
			and
				S   = S_0 (1 - E_1)sin(FA) / (1 - E_1 sin(FA))
				E_1 = exp(-TR/T1)
				FA  = flip angle
				TR  = repetition time

			[1] Tofts, Paul S. "T1-weighted DCE imaging concepts: modelling, acquisition and analysis." signal 500.450 (2010): 400.
		'''
		if FA is None:
			FA = self.FA
		if TR is None:
			TR = self.TR

		self.R1_img     = np.zeros(self.voxelArray.shape)
		E_10            = np.exp(-TR/T_10)
		B               = (1. - E_10)/(1. - E_10*np.cos(FA))
		cosFA           = np.cos(FA)
		minus_1_over_TR = -1./TR
		
		pbar = io.ProgressBar('SI to R1 (slice 1/{})'.format(self.nSlices), self.nSlices)
	
		for sl in range(self.nSlices):
			S   = self.voxelArray[sl]
			S_0 = np.mean(S[:5], axis=0)
			A   = np.nan_to_num(B*S/S_0)
			self.R1_img[sl] = np.nan_to_num(minus_1_over_TR*np.log((1. - A)/(1. - A*cosFA)))
			
			pbar.update(0, 'SI to R1 (slice {}/{})'.format(sl+1, self.nSlices))
		pbar.finish()
		self.R1_img -= np.average(self.R1_img[:5])

		return self.R1_img

class PhantomImage(MRImage):
	def __init__(self, **kwargs):
		
		defaults = {
			'noSlices'      : 1,
			'noRows'        : 4,
			'noCols'        : 4,
			'noTimepoints'  : 100,
			'sectionHeight' : 30,
			'sectionWidth'  : 30,
			'sectionSpacing': 5
		}
		for key in defaults:
			kwargs.setdefault(key, defaults[key])
			setattr(self, key, kwargs[key])

		self.imageHeight = kwargs['noRows']*kwargs['sectionHeight'] + (kwargs['noRows']-1)*kwargs['sectionSpacing']
		self.imageWidth  = kwargs['noCols']*kwargs['sectionWidth']  + (kwargs['noCols'] - 1)*kwargs['sectionSpacing']

		self.voxelArray = np.zeros((kwargs['noSlices'], kwargs['noTimepoints'], self.imageHeight, self.imageWidth))

		self.paramaterImage = None

		MRImage.__init__(self, self.voxelArray)

	def getParameterImage(self, imageIndex=0):
		if self.paramaterImage is None:
			self.paramaterImage = np.zeros((self.noSlices, self.noTimepoints, self.imageHeight, self.imageWidth))
			for slc in range(self.noSlices):
				for i in range(self.noRows):
					for j in range(self.noCols):
						istart = i*self.sectionHeight + i*self.sectionSpacing
						jstart = j*self.sectionWidth  + j*self.sectionSpacing
						self.paramaterImage[slc,0,istart:istart+self.sectionHeight,jstart:jstart+self.sectionWidth] += self.paramOne[j]
						self.paramaterImage[slc,1,istart:istart+self.sectionHeight,jstart:jstart+self.sectionWidth] += self.paramTwo[i]

		return MRImage(self.paramaterImage[:,imageIndex:imageIndex+1,:,:])

	def addNoise(self, std):
		'''
			The noise is here modelled using a gaussian distribution
			with the argument std being the standard deviation of 
			the background noise.
		'''
		if type(std) in [int, float]:
			pbar = io.ProgressBar('Adding noise', self.noRows*self.noCols*self.noSlices)
			for slc in range(self.noSlices):
				for i in range(self.noRows):
					for j in range(self.noCols):
						istart = i*self.sectionHeight + i*self.sectionSpacing
						jstart = j*self.sectionWidth  + j*self.sectionSpacing

						noise = np.random.normal(0, std, (self.noTimepoints, self.sectionHeight, self.sectionWidth))
						self.voxelArray[slc,:,istart:istart+self.sectionHeight,jstart:jstart+self.sectionWidth] += noise
						pbar.update()
			pbar.finish()
		if type(std) == np.ndarray:
			if std.shape[0] != self.noRows or std.shape[1] != self.noCols:
				io.printWarning('The standard deviation (std) array must be the same shape as the phantom (sections). Expected shape ({},{}), got ({}, {}) instead. No noise is added.'.format(self.nRows, self.nCols, std.shape[0], std.shape[1]))
				return
			for slc in range(self.noSlices):
				for i in range(self.noRows):
					for j in range(self.noCols):
						istart = i*self.sectionHeight + i*self.sectionSpacing
						jstart = j*self.sectionWidth  + j*self.sectionSpacing
						
						noise = np.random.normal(0, std[i,j], (self.noTimepoints, self.sectionHeight, self.sectionWidth))
						self.voxelArray[slc,:,istart:istart+self.sectionHeight,jstart:jstart+self.sectionWidth] += noise

	def createFromModel(self, model, **args):
		if model == 'TM':
			self.createFromToftsModel(**args)
		if model == 'ETM':
			self.createFromExtendedToftsModel(**args)
		if model == '2CXM':
			self.createFromTwoCompartmentExchangeModel(**args)

	def createFromToftsModel(self, **args):
		'''
			args:
				firstParamaterRange:
					tuple of length 2 giving the range of paramaters
					used along the first (x) axis of the phanton.
					This defaults to K_trans.
				secondParamaterRange:
					tuple of length 2 giving the range of paramaters
					used along the second (x) axis of the phanton.
					This defaults to k_ep.
				C_a:
					Arterial input function
				t:
					the time array
				axisParamaterNames:
					default: {'first': 'K_trans', 'second': 'k_ep'}
		'''
		args.setdefault('axisParamaterNames', {'first': 'K_trans', 'second': 'k_ep'})
		self.axisParamaterNames = args['axisParamaterNames']
		# loop through the slices
		self.paramOne = np.linspace(args['firstParamaterRange'][0], args['firstParamaterRange'][1], self.noCols)
		self.paramTwo = np.linspace(args['secondParamaterRange'][0], args['secondParamaterRange'][1], self.noRows)

		self.paramOneName = args['axisParamaterNames']['first']
		self.paramTwoName = args['axisParamaterNames']['second']

		TM_params = {
			'C_a': args['C_a'],
			't'  : args['t']
		}
		self.time = args['t']
		self.C_a  = args['C_a']

		# This bit is to find the fixed and set it.
		for p in ['K_trans', 'k_ep', 'v_p', 'v_e']:
			if not p in args['axisParamaterNames'].values() and p in args:
				TM_params[p] = args[p]

		pbar = io.ProgressBar('Creating phantom', self.noRows*self.noCols*self.sectionHeight*self.sectionWidth*self.noSlices)
		for slc in range(self.noSlices):
			# loop through the sections in the phantom
			for i in range(self.noRows):
				TM_params[args['axisParamaterNames']['second']] = self.paramTwo[i]
				for j in range(self.noCols):
					TM_params[args['axisParamaterNames']['first']]  = self.paramOne[j]
					signal = DCE.Models.TM(**TM_params)
					# fill in the section
					istart = i*self.sectionHeight + i*self.sectionSpacing
					jstart = j*self.sectionWidth  + j*self.sectionSpacing
					for ii in range(istart, istart+self.sectionHeight):
						for jj in range(jstart, jstart+self.sectionWidth):
							self.voxelArray[slc,:,ii,jj] = signal
							pbar.update()
		pbar.finish()
		if args['AIFSection']:
			self._addAIFSection()

	def createFromExtendedToftsModel(self, **args):
		'''
			args:
				firstParamaterRange:
					tuple of length 2 giving the range of paramaters
					used along the first (x) axis of the phanton.
					This defaults to K_trans.
				secondParamaterRange:
					tuple of length 2 giving the range of paramaters
					used along the second (x) axis of the phanton.
					This defaults to k_ep.
				C_a:
					Arterial input function
				t:
					the time array
				axisParamaterNames:
					default: {'first': 'K_trans', 'second': 'k_ep'}
		'''
		args.setdefault('axisParamaterNames', {'first': 'K_trans', 'second': 'k_ep'})
		self.axisParamaterNames = args['axisParamaterNames']
		# loop through the slices
		self.paramOne = np.linspace(args['firstParamaterRange'][0], args['firstParamaterRange'][1], self.noCols)
		self.paramTwo = np.linspace(args['secondParamaterRange'][0], args['secondParamaterRange'][1], self.noRows)
		
		self.paramOneName = args['axisParamaterNames']['first']
		self.paramTwoName = args['axisParamaterNames']['second']

		ETM_params = {
			'C_a': args['C_a'],
			't'  : args['t']
		}
		self.time = args['t']
		self.C_a  = args['C_a']

		# ETM requires three parameters to be set. Two are dynamic in the phantom
		# and the third is fixed. This bit is to find the fixed and set it.
		for p in ['K_trans', 'k_ep', 'v_p', 'v_e']:
			if not p in args['axisParamaterNames'].values() and p in args:
				ETM_params[p] = args[p]

		pbar = io.ProgressBar('Creating phantom', self.noRows*self.noCols*self.sectionHeight*self.sectionWidth*self.noSlices)
		for slc in range(self.noSlices):
			# loop through the sections in the phantom
			for i in range(self.noRows):
				ETM_params[args['axisParamaterNames']['second']] = self.paramTwo[i]
				for j in range(self.noCols):
					ETM_params[args['axisParamaterNames']['first']]  = self.paramOne[j]

					signal = DCE.Models.ETM(**ETM_params)
					# fill in the section
					istart = i*self.sectionHeight + i*self.sectionSpacing
					jstart = j*self.sectionWidth  + j*self.sectionSpacing
					for ii in range(istart, istart+self.sectionHeight):
						for jj in range(jstart, jstart+self.sectionWidth):
							self.voxelArray[slc,:,ii,jj] = signal
							pbar.update()
		pbar.finish()
		if args['AIFSection']:
			self._addAIFSection()

	def createFromTwoCompartmentExchangeModel(self, **args):
		'''
			args:
				firstParamaterRange:
					tuple of length 2 giving the range of paramaters
					used along the first (x) axis of the phanton.
					This defaults to K_trans.
				secondParamaterRange:
					tuple of length 2 giving the range of paramaters
					used along the second (x) axis of the phanton.
					This defaults to k_ep.
				C_a:
					Arterial input function
				t:
					the time array
				axisParamaterNames:
					default: {'first': 'PS', 'second': 'F_p'}
		'''
		args.setdefault('AIFSection', True)
		args.setdefault('axisParamaterNames', {'first': 'PS', 'second': 'F_p'})
		self.axisParamaterNames = args['axisParamaterNames']
		# loop through the slices
		self.paramOne = np.linspace(args['firstParamaterRange'][0], args['firstParamaterRange'][1], self.noCols)
		self.paramTwo = np.linspace(args['secondParamaterRange'][0], args['secondParamaterRange'][1], self.noRows)
		
		self.paramOneName = args['axisParamaterNames']['first']
		self.paramTwoName = args['axisParamaterNames']['second']

		twoCXM_params = {
			'C_a': args['C_a'],
			't'  : args['t']
		}
		self.time = args['t']
		self.C_a  = args['C_a']

		# 2CXM requires four parameters to be set. Two are dynamic in the phantom
		# and the third and fourth is fixed. This bit is to find the fixed paramaters and set it.
		for p in ['PS', 'F_p', 'v_p', 'v_e', 'K_trans', 'k_ep']:
			if not p in args['axisParamaterNames'].values() and p in args:
				twoCXM_params[p] = args[p]

		pbar = io.ProgressBar('Creating phantom', self.noRows*self.noCols*self.sectionHeight*self.sectionWidth*self.noSlices)
		for slc in range(self.noSlices):
			# loop through the sections in the phantom
			for i in range(self.noRows):
				twoCXM_params[args['axisParamaterNames']['second']] = self.paramTwo[i]
				for j in range(self.noCols):
					twoCXM_params[args['axisParamaterNames']['first']]  = self.paramOne[j]
					signal = DCE.Models.twoCXM(**twoCXM_params)
					# fill in the section
					istart = i*self.sectionHeight + i*self.sectionSpacing
					jstart = j*self.sectionWidth  + j*self.sectionSpacing
					for ii in range(istart, istart+self.sectionHeight):
						for jj in range(jstart, jstart+self.sectionWidth):
							self.voxelArray[slc,:,ii,jj] = signal
							pbar.update()
		pbar.finish()
		if args['AIFSection']:
			self._addAIFSection()

	def plot(self, **args):
		super(PhantomImage, self).plot(vmin=np.min(self.voxelArray), vmax=np.max(self.voxelArray), xlabel='$\mathrm{Increasing} \ ' + self.paramOneName + ' \\rightarrow$', ylabel='$\leftarrow \mathrm{Increasing} \ ' + self.paramTwoName + '$', **args)

	def _addAIFSection(self):
		self.AIFSectionHeight = int(np.ceil(self.sectionHeight*0.5))
		self.imageHeight += self.AIFSectionHeight + self.sectionSpacing

		voxelArray = np.zeros((self.noSlices, self.noTimepoints, self.imageHeight, self.imageWidth))
		
		start = self.imageHeight - self.AIFSectionHeight
		voxelArray[:,:,:start-self.sectionSpacing,:] = self.voxelArray

		for i in range(len(self.C_a)):
			voxelArray[:,i,start:,:] = self.C_a[i]
		self.voxelArray = voxelArray

	def getAverageTable(self, img):
		'''
			will return the average value of img within each section
			of the phantom, assuming the sections are as in self
		'''
		average = np.zeros((img.shape[0], img.shape[1], self.noRows, self.noCols))
		for slc in range(img.shape[0]):
			for t in range(img.shape[1]):
				for i in range(self.noRows):
					for j in range(self.noCols):
						istart = i*(self.sectionHeight  + self.sectionSpacing)
						jstart = j*(self.sectionWidth  + self.sectionSpacing)
						section = img[slc,t,istart:istart+self.sectionHeight,jstart:jstart+self.sectionWidth]
						average[slc,t,i,j] = np.average(section)
		return average