'''
Some auxilliary functions that provide utilities to the rest of the package
'''

import numpy as np

def check_param_size(**kwargs):
    '''
    For tissue model inputs, we accepts parameter inputs as either
    multi-dimensional arrays or scalar values. If they are arrays they must all be the same size, and are flattened to 1D vectors. If they are scalar
    they are converted to 1-element 1D vectors.

    If any inputs do not match a ValueError is raised

    inputs: keyword list of parameters, either scalars, 1 element arrays or 
        n-d arrays of the same size (but necessarily shape)

    outputs:
        n_vox: size of parameter arrays, 1 iff all parameters are scalars

        params: input parameters reshaped as 1D array
    '''
    n_vox = 0
    for param_name, param in kwargs.items():
        kwargs[param_name] = np.array(param).flatten()
        if kwargs[param_name].size > n_vox:
            n_vox = kwargs[param_name].size

    for param_name, param in kwargs.items():
        print(f'{param_name} size = {param.size}')

        if param.size > 1 and param.size != n_vox:
            raise ValueError(
                f'Error, size of {param_name} ({param.size}) does not match the size of the other parameters ({n_vox})')
    
    return (n_vox, *kwargs.values())
    
def check_param_shape(**kwargs):
    '''
    For parameter conversion, we accepts parameter inputs as either
    multi-dimensional arrays (or array like objects) or scalar values. If they are arrays they must all be the same shape. If they are scalar
    they are converted to np.arrays

    If any inputs do not match a ValueError is raised

    inputs: keyword list of parameters, either scalars or 
        n-d arrays of the same shape

    outputs:
        shape: shape of parameters

        params: input parameters converted to np.arrays
    '''
    shape = None
    
    for param_name, param in kwargs.items():
        kwargs[param_name] = np.array(param)

        if kwargs[param_name].size > 1:          
            if shape is None:
                shape = kwargs[param_name].shape
            elif shape != kwargs[param_name].shape:
                raise ValueError(
                    f'Error, shape of {param_name} ({kwargs[param_name].shape}) does not match the shape of the other parameters ({shape})')

    if shape is None: #all inputs were scalar...
        shape = (1)

    return (shape, *kwargs.values())

#
#-------------------------------------------------------------------------------
def trapz_integral(C_t, t):
    '''
    Compute cumulative integral of time-series at given times using trapezium rule

    inputs:

        C_t : np.array (n_times)
            Time-series to integrate
        t : np.array (n_times)
            Associated time-points

    outputs:
        C_t_integral : np.array (n_times)
            Cumulative integral at each timepoint
    '''
    delta_t = t[1:] - t[0:-1]
    C_t_mid = 0.5*(C_t[0:-1] + C_t[1:])

    C_t_integral = np.zeros_like(t)
    C_t_integral[1:] = np.cumsum(delta_t*C_t_mid)
    return C_t_integral

#
#-------------------------------------------------------------------------------
def exp_conv(T:float, delta_t:float, Ca1:float, Ca0:float, f0:float):
    '''
    Computes update to convolution of function T.exp(-t_i*T) with Ca(t_i) between t_i and t_i-1
    
    inputs:
        T exponent:float parameter
        delta_t:float time difference t_i - t_i-1
        Ca1:float value of Ca at t_i
        Ca0:float value of Ca at t_i-1
        f0:float previous value of convolved function to be updated
    outputs:
        f1:Update convolved function
  */
    '''
    xi = delta_t * T
    delta_a = (Ca1 - Ca0) / xi
    E = np.exp(-xi)
    E0 = 1 - E
    E1 = xi - E0

    integral = Ca0*E0 + delta_a*E1
    f1 = E*f0 + integral
    return f1