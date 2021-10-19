'''
Module to compute signal from T1 for VFA, SPGR

Note we don't do any fitting here, as this module is used primarily to
provide test data in the Madym wrappers
'''

import numpy as np
def signal_from_T1(T1, M0, FA, TR):
    '''
      compute MR signal given T1 and flip angle
       St = signal_from_T1(T1, M0, FA, TR)
    
     Parameters:
          T1 : array_like 
            Input T1. T1, M0 and TR must be scalars or arrays of the same
          size. Arrays will be reshaped to a single column vector
    
          M0 : array_like
            Signal(s) at baseline
    
          FA : array_like
            Flip angle(s) in degrees. Can be input as 2D array T1.size x num_angles
            allowing a different set of flip angles for each (T1[i], S1[i], signal[i])
            sample (though needs to be the same number of FAs), or as a single list
            (or scalar) of FA(s) applied to each signal
    
          TR : array_like
            Recovery time in msecs, either single value for all voxels or array
            matching size of T1 and M0
    
    
     Returns:
          St - MR signal, signal.size x nFAs
     ''' 
    T1 = np.atleast_2d(T1).reshape(-1,1).astype(float) #Set T1, M0 and TR as col vectors
    M0 = np.atleast_2d(M0).reshape(-1,1).astype(float)
    TR = np.atleast_2d(TR).reshape(-1,1).astype(float)
    FA = np.atleast_2d(FA).astype(float) 
    
    #FA as row vector or 2D array, T1.size x n_FAs
    if FA.shape[0] != np.max([T1.size, M0.size, TR.size]):
        FA = FA.reshape(1,-1)

    #Convert degrees to radians
    FA *= np.pi / 180

    #Compute signal, broadcasting to T1.size x FA.size
    E1 = np.exp(-TR / T1)
    St = M0 * np.sin(FA) * (1 - E1) / (1 - np.cos(FA) * E1)
    return St
    
