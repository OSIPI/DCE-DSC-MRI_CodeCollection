'''
Arterial input function class that supports:

* Reading a AIF from text file
* Generating a population AIF using the Parker method
* Passing a vector of AIF values as input
* Resampling the created AIF with any given delay time

Aif times are assumed to be in minutes

'''

import numpy as np
from enum import Enum
from scipy.interpolate import interp1d
import warnings

class AifType(Enum):
    STD = 0 #don't use auto() for CSF compatibility
    FILE = 1
    POP = 2
    ARRAY = 3

class Aif:

    def __init__(self, aif_type:AifType=AifType.POP, filename:str =None,   
        times:np.array=np.empty(0), 
        base_aif:np.array=np.empty(0), 
        base_pif:np.array=np.empty(0),
        prebolus:int=8, hct:float=0.0, dose:float=0.1):
        # Create some member animals
        self.type_ = aif_type
        self.times_ = np.array(times)
        self.base_aif_ = np.array(base_aif)
        self.base_pif_ = np.array(base_pif)
        self.prebolus_ = prebolus
        self.hct_ = hct
        self.dose_ = dose
        self.resampled_AIF_ = np.array([])
        self.resampled_PIF_ = np.array([])
        self.PIF_IRF_ = np.array([])

        if aif_type == AifType.POP:
            if self.base_aif_.size:
                warnings.warn("base_aif set but will be overwritten because aif_type=POP. Use aif_type=ARRAY if setting base_aif")

            if self.num_times() and (self.num_times() < self.prebolus_):
                raise ValueError(f'Size of times {self.num_times()} must be at least prebolus {self.prebolus_}')

            self.base_aif_ = self.compute_population_AIF()

        elif aif_type == AifType.FILE and filename:
            if self.base_aif_.size:
                warnings.warn("base_aif set but will be overwritten because aif_type=FILE and filename is set. Use aif_type=ARRAY if setting base_aif")

            self.read_AIF(filename, not times.size)
 
 
    def num_times(self):
        '''
        Return the number of time points in the Aif
        Parameters:
            None

        Returns:
            n_times : int
                Number of times in the Aif
        '''
        return self.times_.size

    def compute_population_AIF(self, offset=0):
        '''
            Compute a population Aif using the Parker method
            
            Parameters:
                offset : float = 0
                    Delay time at which times_ will be offset

            Returns:
                pop_aif: np.array
                    (n_times,1) array of AIF values

            Notes:
                Requires times_ to have been set

        '''
        #If we don't have times set, do nothing
        if not self.num_times():
            return

        #Compute population AIF using GMP's method
        offset = np.atleast_2d(offset)
        n_v = offset.size
        offset.shape = (n_v,1)

        prebolus_time = self.times_[self.prebolus_-1]
        t_offset = self.times_.reshape(1, self.num_times()) \
                - offset - prebolus_time

        #A1/(SD1*sqrt(2*PI)) * exp(-(t_offset-m1)^2/(2*var1))
        #A1 = 0.833, SD1 = 0.055, m1 = 0.171
        gaussian1 = 5.73258 * np.exp(
            -1.0 *
            (t_offset - 0.17046) * (t_offset - 0.17046) /
            (2.0 * 0.0563 * 0.0563) )
        
        #A2/(SD2*sqrt(2*PI)) * exp(-(t_offset-m2)^2/(2*var2))
        #A2 = 0.336, SD2 = 0.134, m2 = 0.364
        gaussian2 = 0.997356 * np.exp(
            -1.0 *
            (t_offset - 0.365) * (t_offset - 0.365) /
            (2.0 * 0.132 * 0.132))
        # alpha*exp(-beta*t_offset) / (1+exp(-s(t_offset-tau)))
        # alpha = 1.064, beta = 0.166, s = 37.772, tau = 0.482
        sigmoid = 1.050 * np.exp(-0.1685 * t_offset) / (1.0 + np.exp(-38.078 * (t_offset - 0.483)))
        pop_aif = ((self.dose_ / 0.1) * (gaussian1 + gaussian2 + sigmoid)) / \
            (1.0 - self.hct_)
        
        return pop_aif


    def read_AIF(self, filename, use_file_times=True):
        #Read an AIF from file
        aif_data = np.loadtxt(filename)
        self.base_aif_ = aif_data[:,1]
        if use_file_times:
            self.times_ = aif_data[:,0]

    def write_AIF(self, filename):
        #Write base AIF to file
        combined_AIF = np.concatenate(
            (np.reshape(self.times_, (-1,1)), 
            np.reshape(self.base_aif_, (-1,1))), axis=1)
        np.savetxt(filename, combined_AIF)


    def resample_AIF(self, offset):
        if not self.num_times():
            raise RuntimeError("Can't resample AIF until times are set")

        #Resample the AIF given a time offset
        offset = np.atleast_2d(offset)
        n_v = offset.size
        offset.shape = (n_v,1)

        if self.type_ == AifType.POP:
            self.resampled_AIF_ = self.compute_population_AIF(offset)
            return self.resampled_AIF_

        elif self.type_ == AifType.FILE or self.type_ == AifType.ARRAY:
            t_offset = self.times_.reshape(1, self.num_times()) \
                - offset #output is nv x nt

            aif_interp = interp1d(self.times_, self.base_aif_, 
                    kind='linear', fill_value='extrapolate', assume_sorted=True)
            self.resampled_AIF_ = aif_interp(t_offset)
            return self.resampled_AIF_

    def resample_PIF(self, offset, offsetAIF, resampleIRF):
        if not self.num_times():
            raise RuntimeError("Can't resample PIF until times are set")

        #Resample the AIF given a time offset
        n_t = self.num_times()
        offset = np.atleast_2d(offset)
        n_v = offset.size
        offset.shape = (n_v,1)

        #If we've got an offset, make sure AIF has been resampled
        if offsetAIF or self.resampled_AIF_.shape != (n_v,n_t):
            self.resample_AIF(offset) #nv x nt  

	    #generate a population IRF according to Anita's model
        if resampleIRF or (len(self.PIF_IRF_) != n_t) or np.any(np.isnan(self.PIF_IRF_)):
            
            self.PIF_IRF_ = np.zeros((n_v,n_t))
            irf_sum = 0.0
            for i_t in range(n_t):
                t = self.times_[i_t] - offset #(n_v x 1)
                if (t < 0.08):
                    self.PIF_IRF_[:,i_t] = 0#This might have been set to NaN, so make sure we set back to zero
                elif (t < 0.17):
                    self.PIF_IRF_[:,i_t] = 24.16*t - 2.01
                else:
                    self.PIF_IRF_[:,i_t] = 2.83*np.exp(-10.80*t) + 2.12*np.exp(-1.82*t)

                irf_sum += self.PIF_IRF_[:,i_t]

            self.PIF_IRF_ /= irf_sum #Broadcast assignment division? Numpy I love you... ;o)
	
        #Convolve the AIF with the IRF to generate the PIF
        self.resampled_PIF_ = np.zeros((n_v, n_t))

        print('aif: ', self.resampled_AIF_.shape)
        print('pif: ', self.resampled_PIF_.shape)
        print('irf: ', self.PIF_IRF_.shape)

        #literal convolution operation
        for i_t in range(n_t):

            pif_sum = 0
            k_t = i_t
            for j_t in range(i_t + 1):
                pif_sum += self.resampled_AIF_[:,j_t] * self.PIF_IRF_[:,k_t] #(nv x 1)*(nv x 1)
                k_t -= 1     

            self.resampled_PIF_[:,i_t] = pif_sum

        return self.resampled_PIF_

        
            


        
