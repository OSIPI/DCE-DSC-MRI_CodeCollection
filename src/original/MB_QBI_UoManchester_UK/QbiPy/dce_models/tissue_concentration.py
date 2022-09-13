'''
Functions to convert between signal and contrast-agent concentration
'''

import numpy as np

#
#-------------------------------------------------------------------------------
def signal_to_concentration(S_t: np.array, T1_0: np.array, M0: np.array, 
    FA: float, TR: float, relax_coeff: float, use_M0_ratio: int) ->np.array:
    '''
    Convert signal intensity time series to concentrations, based on known imaging and physiological parameters
     Parameters:
       S_t (2D np.array,n_voxels x num_times): Input signal intensity time series
    
       T1_0 (1D np.array, n_voxels): Baseline T1 value associated with each voxel 
    
       M0 (1D np.array, n_voxels): Baseline M0 value associated with each voxel, if use_s0_ratio is positive
       this should be the target pre-contrast signal at each voxel. The scaling factor over the time series
       will then be computed so the mean signal of the first use_s0_ratio time points equals the target signal
       otherwise supply the M0 scaling factor computed with the baseline T1, numpy array, n_voxels
    
       FA (float, default 20.0): flip angle of dynamic images in degrees
    
       TR (float, default 4.0): TR of dynamic images
    
       relax_coeff (float, default 3.4e-3): Relaxivity coefficient of concentration agents (default Ominscan?)
    
       use_M0_ratio (int, default 8): See M0, if a positive integer, defines the number of initial time points to use in an M0 calculation
       if <= 0, baseline M0 values (eg estimated alongside baseline T1) must be supplied
    
    
     Returns:
          C_t (2D numpy array, n_voxels x n_times): concentration time series, 
    
    '''
    #We specify relaxivity in ms, so need to divide by 1000 to get s
    relax_coeff /= 1000

    S_t = np.atleast_2d(S_t)
    if S_t.shape[1]==1:
        S_t = np.transpose(S_t)

    num_voxels = S_t.shape[0]

    T1_0 = np.array(T1_0)
    if T1_0.size != num_voxels:
        #Flag error - throw exception?, return empty signals
        raise ValueError(
            f'Size of T1_0 ({T1_0.size}) does not match number of  rows in S_t ({num_voxels}')
        
    T1_0 = T1_0.reshape(num_voxels,1)

    #Convert FA from degrees to radians
    FA = np.pi * FA / 180
    sin_FA = np.sin(FA)
    cos_FA = np.cos(FA)

    if use_M0_ratio > 0:
        #Compute scaling factor m_0 from S_0 and T1_0, given the fact C_0 = 0
        S_0 = np.mean(S_t[:,:use_M0_ratio,],1).reshape(num_voxels,1)

        e_0 = np.exp(-TR / T1_0)
        a_0 = sin_FA*(1 - e_0)
        b_0 = 1 - cos_FA*e_0
        M0 = S_0 * b_0 / a_0
    else:
        #Check M0 is correct size
        M0 = np.array(M0)
        if M0.size != num_voxels:
            #Flag error - throw exception?, return empty signals
            raise ValueError (f'Size of M0 ({M0.size}) does not match number of columns in S_t ({num_voxels})')
        M0 = M0.reshape(num_voxels,1)
    
    #Use broadcasting to scale input signal by M0
    St_hat = S_t / M0
    e_t = (sin_FA - St_hat) / (sin_FA - St_hat*cos_FA)
    #e_t[e_t < 0] = np.min(e_t[e_t >= 0])

    R1_t = -np.log(e_t) / TR

    C_t = (R1_t - 1 / T1_0) / relax_coeff
    return C_t

#
#-------------------------------------------------------------------------------
def concentration_to_signal(C_t: np.array, T1_0: np.array, M0: np.array, 
    FA: float, TR: float, relax_coeff: float, use_M0_ratio: int = 8)->np.array:
    '''
     Convert concentration time series to signal intensities, based on known imaging and physiological parameters
     Parameters:
       C_t (2D np.array,n_voxels x num_times): Input concentration time series
    
       T1_0 (1D np.array, n_voxels): Baseline T1 value associated with each voxel  
    
       M0 (1D np.array, n_voxels): Baseline M0 value associated with each voxel, if use_s0_ratio is positive
       this should be the target pre-contrast signal at each voxel. The scaling factor over the time series
       will then be computed so the mean signal of the first use_s0_ratio time points equals the target signal
       otherwise supply the M0 scaling factor computed with the baseline T1
    
       FA (float, default 20.0): flip angle of dynamic images in degrees
    
       TR (float, default 4.0): TR of dynamic images
    
       relax_coeff (float, default 3.4): Relaxivity coefficient of concentration agents (default Ominscan?) in ms
    
       use_M0_ratio (int, default 8): See M0, if a positive integer, defines the number of initial time points to use in an M0 calculation
       if <= 0, baseline M0 values (eg estimated alongside baseline T1) must be supplied
    
    
     Returns:
          S_t (2D numpy array, n_voxels x n_times): signal time series, n_voxels x num_times 
    ''' 

    #We specify relaxivity in ms, so need to divide by 1000 to get s
    relax_coeff /= 1000

    #Make sure signal inputs are all 1 x m
    C_t = np.atleast_2d(C_t)
    if C_t.shape[1]==1:
        C_t = np.transpose(C_t)

    num_voxels = C_t.shape[0]

    T1_0 = np.array(T1_0)
    if T1_0.size != num_voxels:
        #Flag error - throw exception?, return empty signals
        raise ValueError (
            f'Size of T1_0 ({T1_0.size}) does not match number of rows in C_t ({num_voxels})')

    M0 = np.array(M0)
    if M0.size != num_voxels:
        #Flag error - throw exception?, return empty signals
        raise ValueError (
            f'Size of M0 ({M0.size}) does not match number of rows in C_t ({num_voxels})')

    T1_0 = T1_0.reshape(num_voxels,1)
    M0 = M0.reshape(num_voxels,1)

    #R1 is 1/T1
    R1_t = relax_coeff*C_t + 1/T1_0.reshape(num_voxels,1)

    #Convert FA from degrees to radians
    FA = np.pi * FA / 180

    #Apply the concentration calculations
    e_t = np.exp(-TR * R1_t)
    a_t = np.sin(FA)*(1 - e_t)
    b_t = 1 - np.cos(FA)*e_t
    St_hat = a_t / b_t

    #If computing M0 using the ratio method, take the mean
    #of the scaled sign
    if use_M0_ratio > 0:
        M0_hat = np.mean(St_hat[:,:use_M0_ratio], 1).reshape(num_voxels,1)
        M0 = M0.reshape(num_voxels,1) / M0_hat

    #Make use of numpy broadcasting, St_hat is (num_voxels, num_times), M0 is (num_voxels,1)
    S_t = St_hat * M0
    return S_t

#
#-------------------------------------------------------------------------------
def compute_IAUC(C_t: np.array, dyn_times: np.array, aif_injection_image: int = 8,
    interval: float = 60.0, time_scaling: float = 1)->float:
    '''
    Compute area under concentration curve for given number of seconds
    
     Parameters:
       C_t (nD np.array, n1 x ... x ni x n_times ): Input concentration time series, can be
       multidimensional, last dimension is time (this allows us to call same function on extracted
       voxel data, 2D images, 3D volumes etc)
    
       dyn_times (1D np.array, n_times): time (in seconds or minutes)
    
       aif_injection_image (int >= 0, default 8) - index of image at point bolus was injected
    
       interval (float): time in seconds from start of time series
    
       time_scaling (float, default 1.0): scaling factor applied to times, eg to convert from mins  (as used in tofts model) to seconds
    '''
    n_dims = C_t.ndim
    n_times = C_t.shape[-1]

    #Check size of dyn_times input and scale
    if dyn_times.size != n_times:
        raise ValueError(
            f'Length of dyn_times ({dyn_times.size}) does not much number of volumes in C_t array ({n_times})')

    if dyn_times.ndim > 1:
        dyn_times = dyn_times.reshape(n_times)
    
    dyn_times *= time_scaling

    #Compute index of image to sum up to (since we're linearly interpolating, why 
    #stop at an integer image?)
    iauc_idx = np.nonzero((dyn_times - dyn_times[aif_injection_image-1]) > interval)[0][0]

    #Take difference of times to get time intervals
    time_intervals = dyn_times[1:iauc_idx] - dyn_times[0:iauc_idx-1]

    #We can make use of broadcasting (and ...) here to multiply last dimension of
    #concentration array with the time intervals vector, then sum along this dimension
    C_t_avg = (0.5*C_t[...,0:iauc_idx-1] +  0.5*C_t[...,1:iauc_idx])          
    iauc = np.sum( C_t_avg * time_intervals, n_dims-1)
    return iauc