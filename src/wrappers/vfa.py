import osipi_code_collection.original.MJT_UoEdinburghUK.t1_fit as edinburgh
import osipi_code_collection.original.ST_USydAUS.VFAT1mapping as sydney
import osipi_code_collection.original.McGill_Can.vfa as mcgill
from osipi_code_collection.utils.nb import percenterror
import matplotlib.pyplot as plt
import numpy as np

def vfa_fit(author, signal = None, fa = None, tr = None, fittype = "linear", mask = None, data = None):
    # author could be: "edinburgh", "sydney", "mcgill"
    # signal: numpy array, last dimension must be flip angle
    # fa: flip angles, in radian
    # tr: repetition time, in ms
    # alternatively, `data` can be input containing the above three elements


    # Unpack variables if data object is given
    if not data == None:
        signal = data.signal
        fa = data.fa
        tr = data.TR
    
    # Some implementations only work if input signal is 1D
    requires_1D = author in ["edinburgh", "sydney"]

    # Reshape input as Voxels-by-FlipAngle matrix
    spatialdims = signal.shape[:-1]
    numvox = np.prod(spatialdims)
    numangles = signal.shape[-1]
    signal = signal.reshape(-1, numangles)
    if mask != None:
        signal = signal[mask[:] > 0, :]

    if author == "edinburgh":
        fa = np.rad2deg(fa)
        if fittype == "nonlinear":
            fitfunc = lambda S, FA, TR: edinburgh.fit_vfa_nonlinear(S, FA, TR)
        else:
            fitfunc = lambda S, FA, TR: edinburgh.fit_vfa_linear(S, FA, TR)
    elif author == "sydney":
        fa = np.rad2deg(fa)
        fitfunc = lambda S, FA, TR: sydney.VFAT1mapping(FA, S, TR, method = fittype)
    elif author == "mcgill":
        if fittype == "nonlinear":
           fitfunc = lambda S, FA, TR: mcgill.novifast(S, FA, TR)
        elif fittype == "nonlinear_noniterative":
            fitfunc = lambda S, FA, TR: mcgill.novifast(S, FA, TR, doiterative = False)
        else: # linear
            fitfunc = lambda S, FA, TR: mcgill.despot(S, FA, TR)
    else:
        print("ERROR: Unexpected author")
        return

    if requires_1D:
        _M0 = np.zeros(numvox)
        _T1 = np.zeros(numvox)
        for idx in range(numvox):
            _M0[idx], _T1[idx] = fitfunc(signal[idx, :], fa, tr)
    else:
        _M0, _T1 = fitfunc(signal, fa, tr)
    
    if mask != None:
        M0 = np.zeros(spatialdims)
        T1 = np.zeros(spatialdims)
        M0[mask[:] > 0] = _M0
        T1[mask[:] > 0] = _T1
    else:
        M0 = _M0.reshape(spatialdims)
        T1 = _T1.reshape(spatialdims)
    return (M0, T1)
  
def show_maps(fittedmaps = None, M0 = None, T1 = None, truth = None, returnfig = False):
    if not np.any(M0):
        M0, T1 = fittedmaps

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
    ax1.imshow(M0)
    ax1.grid(False)
    plt.sca(ax1)
    plt.title("M0")
    if np.any(truth):
      xticklocs, yticklocs = get_ticklocs(M0)
      plt.xticks(xticklocs, np.round((np.unique(truth.T1))).astype(int), rotation=45, ha="right")
      plt.yticks(yticklocs, np.round((np.unique(truth.M0))).astype(int))
    
    ax2.imshow(T1)
    ax2.grid(False)
    plt.sca(ax2)
    plt.title("T1")
    if np.any(truth):
      xticklocs, yticklocs = get_ticklocs(M0)
      plt.xticks(xticklocs, np.round((np.unique(truth.T1))).astype(int), rotation=45, ha="right")
      plt.yticks(yticklocs, np.round((np.unique(truth.M0))).astype(int))
    
    # fig.tight_layout()
    if returnfig:
        return fig
    return

def show_error_maps(fits, idx = -1, title = "", showcbar = False, truth = None, clim = (-100, 100), returnfig = False):
    M0, T1 = fits

    if (idx >= 0):
        M0 = M0[:,:,idx]
        T1 = T1[:,:,idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
    ax1.imshow(percenterror(M0, truth.M0), cmap = "PuOr_r", clim = clim)
    ax1.grid(False)
    plt.sca(ax1)
    plt.title("%error in M0")
    if np.any(truth):
      xticklocs, yticklocs = get_ticklocs(M0)
      plt.xticks(xticklocs, np.round((np.unique(truth.T1))).astype(int), rotation=45, ha="right")
      plt.yticks(yticklocs, np.round((np.unique(truth.M0))).astype(int))
    
    im = ax2.imshow(percenterror(T1, truth.T1), cmap = "PuOr_r", clim = clim)
    ax2.grid(False)
    plt.sca(ax2)
    plt.title("%error in T1")
    if np.any(truth):
      xticklocs, yticklocs = get_ticklocs(M0)
      plt.xticks(xticklocs, np.round((np.unique(truth.T1))).astype(int), rotation=45, ha="right")
      plt.yticks(yticklocs, np.round((np.unique(truth.M0))).astype(int))
    
    fig.suptitle(title)
    if showcbar:
        fig.colorbar(im, ax=[ax1,ax2], orientation='horizontal', shrink = 0.5)
    # else:
    #     fig.tight_layout()

    if returnfig:
        return fig
    return

def get_ticklocs(arr):
    if arr.shape[0] > 20:
        # Full resolution maps - 70x150
        xticklocs = np.arange(5, 150, 10)
        yticklocs = np.arange(5, 70, 10)
    else:
        # Downsampled maps - 7x15
        xticklocs = np.arange(0, 15, 1)
        yticklocs = np.arange(0, 7, 1)
    return xticklocs, yticklocs