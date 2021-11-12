import ..original.MJT_UoEdinburghUK.t1_fit as edinburgh
import ..original.ST_USydAUS.VFAT1mapping as sydney
import ..original.McGill_Can.vfa as mcgill
import matplotlib.pyplot as plt

def vfa_fit(_signal, fa, tr, author, fittype = "linear", mask = None):
    # Signal: numpy array, last dimension must be flip angle
    # fa: flip angles, in radian
    # tr: repetition time, in ms
    
    spatialdims = signal.shape[:-1]
    signal = signal.reshape(-1, signal.shape[-1])
    if mask != None:
        signal = signal[mask[:] > 0, :]
        
    if author == "edinburgh":
        if fittype == "nonlinear":
            _M0, _T1 = edinburgh.fit_vfa_nonlinear(signal, fa, tr)
        else: # linear
            _M0, _T1 = edinburgh.fit_vfa_linear(signal, fa, tr)
    elif author == "sydney":
        fa = np.rad2deg(fa)
        tr = tr
        numvox = np.prod(spatialdims)
        _M0 = np.zeros(numvox)
        _T1 = np.zeros(numvox)
        for idx in range(numvox):
            _M0[idx], _T1[idx] = sydney.VFAT1mapping(fa, signal[idx, :], tr, method = fittype)
    elif author == "mcgill":
        if fittype == "nonlinear":
            _M0, _T1 = mcgill.novifast(signal, fa, tr)
        elif fittype == "nonlinear_noniterative":
            _M0, _T1 = mcgill.novifast(signal, fa, tr, doiterative = False)
        else: # linear
            _M0, _T1 = mcgill.despot(signal, fa, tr)
    else:
        print("ERROR: Unexpected author")
        return
    
    if mask != None:
        M0 = np.zeros(spatialdims)
        T1 = np.zeros(spatialdims)
        M0[mask[:] > 0] = _M0
        T1[mask[:] > 0] = _T1
    else:
        M0 = _M0.reshape(spatialdims)
        T1 = _T1.reshape(spatialdims)
    return (M0, T1)
  
def show_maps(fittedmaps = None, M0 = None, T1 = None, truth = None):
    if not np.any(M0):
        M0, T1 = fittedmaps
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
    ax1.imshow(M0)
    ax1.grid(False)
    plt.sca(ax1)
    plt.title("M0")
    if np.any(truth):
      plt.xticks(np.arange(5, 150, 10), np.round((np.unique(truth.T1))).astype(int), rotation=45, ha="right")
      plt.yticks(np.arange(5, 70, 10), np.round((np.unique(truth.M0))).astype(int))
    
    ax2.imshow(T1)
    ax2.grid(False)
    plt.sca(ax2)
    plt.title("T1")
    if np.any(truth):
      plt.xticks(np.arange(5, 150, 10), np.round((np.unique(truthdata.T1))).astype(int), rotation=45, ha="right")
      plt.yticks(np.arange(5, 70, 10), np.round((np.unique(truthdata.M0))).astype(int))
    
    fig.tight_layout()
    return

def show_error_maps(fits, title = "", showcbar = False, truth = None, clim = (-100, 100)):
    M0, T1 = fits
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
    ax1.imshow(percenterror(M0, truth.M0), cmap = "PuOr_r", clim = clim)
    ax1.grid(False)
    plt.sca(ax1)
    plt.title("%error in M0")
    plt.xticks(np.arange(5, 150, 10), np.round((np.unique(truthdata.T1))).astype(int), rotation=45, ha="right")
    plt.yticks(np.arange(5, 70, 10), np.round((np.unique(truthdata.M0))).astype(int))
    
    im = ax2.imshow(percenterror(T1, truth.T1), cmap = "PuOr_r", clim = clim)
    ax2.grid(False)
    plt.sca(ax2)
    plt.title("%error in T1")
    plt.xticks(np.arange(5, 150, 10), np.round((np.unique(truthdata.T1))).astype(int), rotation=45, ha="right")
    plt.yticks(np.arange(5, 70, 10), np.round((np.unique(truthdata.M0))).astype(int))
    
    fig.suptitle(title)
    if showcbar:
        fig.colorbar(im, ax=[ax1,ax2], orientation='horizontal', shrink = 0.5)
    else:
        fig.tight_layout()
    return