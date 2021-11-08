# -*- coding: utf-8 -*-
'''
Created on Thu Aug 19 17:46:47 2021
@author: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

WORK IN PROGRESS
'''

import nibabel as nib
import numpy as np
from MJT_UoEdinburghUK import t1_fit

def fit_vfa_image(tr, fa_rad, filenames, threshold=0, output_name='./vfa_map', method='linear'):
    # Load input images
    images = [nib.load(fn) for fn in filenames]

    # Get and reshape voxel data
    s_nd = np.stack( [img.get_fdata() for img in images], axis=3 ) # N-D signal array
    s_2d = s_nd.reshape(-1, s_nd.shape[-1]) # N-voxels x N-flip-angles

    # Get (T1, S0) for each voxel
    if method=='linear':
        results = [ t1_fit.fit_vfa_linear(s, fa_rad, tr) if np.max(s)>threshold else (np.nan, np.nan) for s in s_2d]
    elif method=='non-linear':
        results = [ t1_fit.fit_vfa_nonlinear(s, fa_rad, tr) if np.max(s)>threshold else (np.nan, np.nan) for s in s_2d]
    else:
        raise ValueError(f'Value of argument method not recognised: {method}')

    # Separate T1 and S0, then reshape to N-D
    s0_1d, t1_1d = zip(*results)
    s0_nd = np.asarray(s0_1d).reshape(s_nd.shape[:-1])
    t1_nd = np.asarray(t1_1d).reshape(s_nd.shape[:-1])
    
    # Write output images
    del s_nd, s_2d, s0_1d, t1_1d
    new_hdr = images[0].header.copy()
    new_hdr.set_data_dtype(np.float32)
    t1_img = nib.nifti1.Nifti1Image(t1_nd, None, header = new_hdr)
    s0_img = nib.nifti1.Nifti1Image(s0_nd, None, header = new_hdr)
    nib.save(t1_img, f'{output_name}_t1.nii')
    nib.save(s0_img, f'{output_name}_s0.nii')
