"""Functions for dealing with images.

Created 6 October 2021
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Functions:
    read_images
    write_image
    roi_measure
"""

import nibabel as nib
import numpy as np


def read_images(images):
    """Read and combine array or nifti images.

    Read one or more images. If there are >1 images, they are concatenated
    along a new dimension.

    Args:
        images (list, str, ndarray): Array containing input image data or str
            corresponding to nifti image file path. If a list of ndarray or str
            is provided the images will first be concatenated along a new
            dimension.

    Returns:
        tuple: (data, header)
            data (ndarray): combined image data
            header (nibabel header): image header of first nifti image. If
                input images are not nifti, None is returned.
    """
    images = [images] if type(images) is not list else images
    if all([type(i) is str for i in images]):
        data = np.stack([nib.load(i).get_fdata() for i in images], axis=-1)
        header = nib.load(images[0]).header
    elif all([type(i) is np.ndarray for i in images]):
        data = np.stack(images, axis=-1)
        header = None
    else:
        raise TypeError(
            "Argument images should contain all strings or all" " ndarrays."
        )
    if data.shape[-1] == 1:
        data = data.squeeze(axis=-1)
    return data, header


def write_image(data, filepath, hdr):
    """Wrapper to save image using nibabel

    Args:
        data (ndarray): image data
        filepath (str): path and filename for output image
        hdr (nibabel header): header template for output image
    """
    img = nib.nifti1.Nifti1Image(data, None, header=hdr)
    nib.save(img, filepath)


def roi_measure(image, mask_image):
    """Calculate statistics for voxels within a mask.

    If the image has the same shape as the mask image, a single set of
    statistics is returned. If the image has one additional dimension (
    e.g. a time series) then a series of values are returned corresponding to
    locations in the last dimension.

    Args:
        image (list, str, ndarray): Array containing input image data or str
            corresponding to nifti image file path. If a list of ndarray or str
            is provided the images will first be concatenated along a new
            dimension.
        mask_image (str, ndarray): Mask image.

    Returns:
        dict{'mean': mean, 'median': median, 'sd': sd}
            mean, median and sd (float, ndarray): statistics for masked
            voxels. For input data with one more dimension than the mask
            image (e.g. a time series), a 1D array of floats is returned.
    """
    # read images and mask
    data, _hdr = read_images(image)
    mask, _hdr = read_images(mask_image)
    if mask.ndim == data.ndim:
        data = np.expand_dims(data, axis=-1)
    if not np.all((mask[:] == 0) | (mask[:] == 1)):
        raise ValueError("Mask contains values that are not 0 or 1.")

    # flatten spatial dimensions
    data_2d = data.reshape(-1, data.shape[-1])  # 2D [location, time] format
    mask_1d = mask.reshape(-1)

    # measure statistics for masked voxels
    masked_voxels = data_2d[mask_1d == 1, :]
    stats = [
        (np.nanmean(m_d), np.nanmedian(m_d), np.nanstd(m_d))
        for m_d in masked_voxels.transpose()
    ]
    mean, median, sd = zip(*stats)

    return {
        "mean": np.squeeze(mean),
        "median": np.squeeze(median),
        "sd": np.squeeze(sd),
    }
