"""Base class Fitter for DCE, T1 and other types of fitting.

Created on Thu Oct 21 15:50:47 2021
@authors: Michael Thrippleton
@email: m.j.thrippleton@ed.ac.uk
@institution: University of Edinburgh, UK

Classes:
    Fitter (abstract base class)
"""

from abc import ABC, abstractmethod
import os

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed

from utils.imaging import read_images, write_image


class Fitter(ABC):
    """Abstract base class for fitting algorithms.

    Subclasses must implement the proc method, which process a single data
    series, e.g. a DCE time series for one voxel/ROI, or a set of signals for
    different flip angles for a T1 measurement. The proc_image method is
    provided for processing images by calling the subclass proc method on each
    voxel.
    """

    @abstractmethod
    def proc(self, *args):
        """Abstract method processing a single data series.

        For example, estimating pharmacokinetic parameters from a DCE
        concentration-time series, or estimating T1 from a series of
        signals acquired at different flip angles. Overridden by subclass.

        Args:
            *args: First argument is the input data, followed by any other
            arguments.

        Returns:
            float or tuple: Output parameter(s).
            If there are >1 outputs, a tuple is returned containing the
            output parameters, each of which should either be a scalar (e.g.
            KTrans) or a 1D array (e.g. fitted concentration series).
        """
        pass

    @abstractmethod
    def output_info(self):
        """Abstract method returning output names and types.

        Returns:
            tuple: name and type of outputs from fitting
            each element is a tuple (str, bool) corresponding to an
            output parameter. str = parameter name. bool = True if
            parameter is 1D (e.g. a time series), False if parameter is a
            scalar (e.g. KTrans).
        """
        pass

    def proc_image(self, input_images, arg_images=None, mask=None,
                   threshold=-np.inf, dir=".", prefix="", suffix="",
                   filters=None, template=None, n_procs=1):
        """Process image voxel-by-voxel using subclass proc method.

        Args:
            input_images (list, str, ndarray): ndarray containing input image
                data or str corresponding to nifti image file path. The last
                dimension is assumed to be the series dimension (e.g. time,
                flip angle). If a list of ndarray or str is provided these
                images will first be concatenated to form a new series
                dimension.
            arg_images (tuple): Tuple containing one image (str or ndarray,
                as above) for each argument needed by the subclass proc method.
                Refer to the subclass proc docstring for required arguments.
                Defaults to None.
            mask (str or ndarray): Mask image (str or ndarray, as above). Must
                contain 1 or 0 only. 1 indicates voxels to be processed.
                Defaults to None (process all voxels).
            threshold (float): Voxel is processed if max input value in
                series (e.g. flip angle or time series) is >= threshold.
                Defaults to -np.inf
            dir (str): Directory for output images. If None, no output
                images are written. Defaults to None.
            prefix (str): filename prefix for output images. Defaults to "".
            suffix (str): filename suffix for output images. Defaults to "".
            filters (dict): Dict of 2-tuples: key=parameter name, value=(lower
                limit, upper limit). Output values outside the range are set
                to nan.
            template (str): Nifti filename. Uses the header of this image to
                write output images. Defaults to None, in which case the header
                of the first input image will be used, otherwise an exception is
                raised.
            n_procs (int): Number of processes for parallel computation.

        Returns:
            array or tuple: Output image(s).
            If there are >1 outputs, a tuple is returned containing
            arrays corresponding to the outputs (e.g. KTrans, ve,
            Ct_fit).
        """

        # read source images, e.g. signal-time images
        data, input_header = read_images(input_images)
        # reshape data to 2D array n_voxels x n_points (length of series)
        data_2d = data.reshape(-1, data.shape[-1])
        n_voxels, n_points = data_2d.shape
        data_shape = data.shape

        names, _ = zip(*self.output_info())

        # read argument images, e.g. flip angle correction, T10
        if arg_images is None:
            args_1d = [()] * n_voxels
        else:
            # get list of N-D arrays for each argument
            arg_arrays, _hdrs = zip(*[
                read_images(a) if type(a) is not float else
                (np.tile(a, data_shape[:-1]), None) for a in arg_images])
            # convert to N-D arrays to list of (list of arguments) per voxel
            args_1d = list(zip(*[a.reshape(-1) for a in arg_arrays]))
            del arg_arrays
        del arg_images

        # read mask image if provided
        if mask is None:
            mask_1d = np.empty(n_voxels, dtype=bool)
            mask_1d[:] = True
        else:
            mask_nd, _ = read_images(mask)
            mask_1d = mask_nd.reshape(-1)
            if any((mask_1d != 0) & (mask_1d != 1)):
                raise ValueError('Mask contains elements that are not 0 or 1.')
            mask_1d = mask_1d.astype(bool)

        # divide data into 1+ "chunks" of voxels for parallel processing
        n_chunks = min(5 * n_procs, n_voxels)
        chunks_start_idx = np.int32(
            n_voxels * (np.array(range(n_chunks)) / n_chunks))

        # function to process a single chunk of voxels (to be called by joblib)
        def _proc_chunk(i_chunk):
            # work out voxel indices corresponding to the chunk
            start_voxel = chunks_start_idx[i_chunk]
            stop_voxel = chunks_start_idx[i_chunk + 1] if (
                    i_chunk != n_chunks - 1) else n_voxels
            n_chunk_voxels = stop_voxel - start_voxel
            # preallocate output arrays
            chunk_output = {}
            for name, is1d in self.output_info():
                n_values = n_points if is1d else 1
                chunk_output[name] = np.empty((n_chunk_voxels, n_values),
                                              dtype=np.float32)
                chunk_output[name][:] = np.nan
            # process all voxels in the chunk
            for i_vox_chunk, i_vox in enumerate(np.arange(start_voxel,
                                                          stop_voxel, 1)):
                voxel_data = data_2d[i_vox, :]
                if max(voxel_data) >= threshold and mask_1d[i_vox]:
                    try:
                        voxel_output = self.proc(voxel_data, *args_1d[i_vox])
                        if len(names) == 1:
                            voxel_output = (voxel_output,)
                        for idx, values in enumerate(voxel_output):
                            chunk_output[names[idx]][i_vox_chunk, :] = values
                    except (ValueError, ArithmeticError):
                        pass  # outputs remain as nan
            return chunk_output

        # run the processing using joblib
        chunk_outputs = Parallel(n_jobs=n_procs)(
            delayed(_proc_chunk)(i_chunk) for i_chunk in range(n_chunks))
        del data, data_2d, args_1d, mask_1d

        # Combine chunks into single output dict
        outputs = {name: np.concatenate(
            [co[name] for co in chunk_outputs], axis=0
        )
            for name, is1d_ in self.output_info()}
        del chunk_outputs

        # filter outputs
        if filters is not None:
            for name, limits in filters.items():
                outputs[name][
                    ~((limits[0] <= outputs[name]) &
                      (outputs[name] <= limits[1]))] = np.nan

        # reshape output arrays to match image shape
        for name, values in outputs.items():
            outputs[name] = np.squeeze(outputs[name].reshape(
                (*data_shape[:-1], values.shape[-1])))

        # write outputs as images if required
        if dir is not None:
            if not os.path.isdir(dir):
                os.mkdir(dir)
            if template is not None:
                hdr = nib.load(template).header
            elif input_header is not None:
                hdr = input_header
            else:
                raise ValueError("Need input nifti files or template nifti "
                                 "file to write output images.")
            hdr.set_data_dtype(np.float32)
            for name, values in outputs.items():
                write_image(outputs[name],
                            os.path.join(dir, f"{prefix}{name}{suffix}.nii"),
                            hdr)

        return outputs[names[0]] if len(names) == 1 else tuple(outputs[name]
                                                               for name in
                                                               names)
