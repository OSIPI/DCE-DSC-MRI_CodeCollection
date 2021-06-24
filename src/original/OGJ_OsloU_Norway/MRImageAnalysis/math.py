from . import io
import numpy as np

class NP:
	'''
		Class with modified numpy functions to make code more dense
	'''
	@staticmethod
	def convolve(x, y, t):
		return (np.convolve(x, y)*(t[1]-t[0]))[:len(t)]

class Integration:
	@staticmethod
	def cumulativeIntegral(x, y, method='trapezoidal'):
		return getattr(Integration, 'cumulative' + method.capitalize())(x, y)

	@staticmethod
	def cumulativeTrapezoidal(x, y):
		ret = np.zeros(len(x))
		ret[1:] = np.cumsum(0.5*(y[:-1] + y[1:])*(x[1:] - x[:-1]))
		return ret

class misc:
	@staticmethod
	def downSample(signal, old_time, new_time):
			'''
				Function to downsample a signal with timepoints
				old_time, into a new signal with timepoints new_time
			'''
			# if new_time[-1] > old_time[-1] or new_time[0] < old_time[0]:
			# 	io.printWarning('New time must be a subset of old time.')
				# return signal

			new_signal = np.zeros(len(new_time))
			for i in range(len(new_time)):
				time_diff = old_time - new_time[i]
				j = np.argmin(abs(time_diff))
				if time_diff[j] < 0:
					j1 = j-1
					j2 = j
				else:
					j1 = j
					j2 = j+1
					if j2 >= len(old_time):
						new_signal[i] = signal[-1]
						continue
				a = (signal[j2] - signal[j1])/(old_time[j2] - old_time[j1])
				b = -a*old_time[j1] + signal[j1]

				new_signal[i] = a*new_time[i] + b

			return new_signal

	def downSampleAverage(old_t, old_signal, dt):
		'''
			Downsaples a signal by first creating a new time array
			with resolution dt, then looping though the indeces
			of the new array and for each time point, calculating the
			average of old_signal from new_t[i]-dt/2 to new_t[i]+dt/2.
			The averaging is upward inclusive and downward exclusive.
		'''
		if old_t[1]-old_t[0] == dt or dt == 0:
			return old_t, old_signal
		new_t = np.arange(0, old_t[-1]*1.001, dt) # times by 1.001 to just include the upper bound
		new_signal = np.zeros(len(new_t))

		first_idx = np.argmin(abs(old_t - dt/2))+1
		new_signal[0] = np.average(old_signal[:first_idx])
		for i in range(1,len(new_t)):
			mid_idx = np.argmin(abs(old_t - i*dt))
			lower_idx = np.argmin(abs(old_t - (old_t[mid_idx]-dt/2)))+1
			upper_idx = np.argmin(abs(old_t - (old_t[mid_idx]+dt/2)))+1
			new_signal[i] = np.average(old_signal[lower_idx:upper_idx])
		new_signal[0] = 0
		return new_t, new_signal