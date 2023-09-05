# Non-osipi dependencies
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter, attrgetter
from scipy.optimize import curve_fit
# osipi utilities
from osipi_code_collection.utils.nb import percenterror
# osipi implementations
import osipi_code_collection.original.LEK_UoEdinburgh_UK.PharmacokineticModelling.models as edinburgh1
import osipi_code_collection.original.MJT_UoEdinburgh_UK.aifs as edinburgh2_aifs
import osipi_code_collection.original.MJT_UoEdinburgh_UK.pk_models as edinburgh2_models
import osipi_code_collection.original.MJT_UoEdinburgh_UK.dce_fit as edinburgh2_fit
import osipi_code_collection.original.OGJ_OsloU_NOR.MRImageAnalysis.DCE.Analyze as oslo
import osipi_code_collection.original.ST_USydAUS.ModelDictionary as sydney
from osipi_code_collection.original.MB_QBI_UoManchester_UK.QbiPy.dce_models import dce_aif as manchester_aif
from osipi_code_collection.original.MB_QBI_UoManchester_UK.QbiPy.dce_models import tofts_model as manchester_tofts
import osipi_code_collection.original.OG_MO_AUMC_ICR_RMH_NL_UK.ExtendedTofts.DCE as amsterdam

def tofts_fit(author, ct = None, ca = None, t = None, fittype = "linear", mask = None, data = None):
  # Author can be: edinburgh1, edinburgh2, sydney, amsterdam, oslo, manchester
  # ct: numpy array containing concentration in tissue of interest, last dimension must be time
  # ca: arterial input function
  # t:  time, in minutes
  # alternatively, `data` can be input containing the above three elements

  # Unpack variables if data object is given
  if not data == None:
    ct = data.ct
    ca = data.ca
    t = data.t

  # Some implementations want time in seconds instead of minutes
  if author in ["edinburgh2"]:
    t = t * 60

  # Some implementations can only fit a voxel at a time (i.e. all inputs must be 1-dimensional)
  requires_1D = author in ["edinburgh1", "edinburgh2", "sydney", "manchester"]

  # Bounds for ktrans and ve for all non-linear approaches (if they require starting point)
  X0 = (0.01, 0.01)
  bounds = ((0.0, 0.0), (1, 1))

  # Convert input to 2D array; NumVoxels-by-NumTimepoints
  spatialdims = ct.shape[:-1]
  numvox = np.prod(spatialdims)
  numtimepoints = ct.shape[-1]
  ct = ct.reshape(-1, numtimepoints)

  # Create a fitting function for the chosen author/implementation
  if author == "edinburgh1":
    fitfunc = lambda C: curve_fit(lambda T, ktrans, ve: edinburgh1.Kety([ktrans, ve], t, ca, 0), t, C, p0=X0, bounds=bounds)[0]
  elif author == "edinburgh2":
    aif = edinburgh2_aifs.patient_specific(t, ca)
    pk_model = edinburgh2_models.tofts(t, aif)
    fitfunc = lambda C: itemgetter('ktrans', 've')(edinburgh2_fit.conc_to_pkp(C, pk_model)[0])
  elif author == "oslo":
    if fittype == "nonlinear":
      fitfunc = lambda C: [x[0,0,:,0] for x in attrgetter("K_trans", "v_e")(oslo.fitToModel('TM', np.array([C.transpose()[:,:,np.newaxis]]), t, ca, showPbar=False, method='NLLS'))]
    else:
      fitfunc = lambda C: [x[0,0,:,0] for x in attrgetter("K_trans", "v_e")(oslo.fitToModel('TM', np.array([C.transpose()[:,:,np.newaxis]]), t, ca, showPbar=False, integrationMethod='trapezoidal', method='LLSQ'))]
  elif author == "sydney":
    time_and_aif = np.column_stack((t,ca))
    # Output is [ve, kt], so we need a switcharoo at the end
    fitfunc = lambda C: curve_fit(sydney.Tofts, time_and_aif, C, p0=X0, bounds=bounds)[0][::-1]
  elif author == "manchester":
    aif = manchester_aif.Aif(times=t, base_aif=ca, aif_type =  manchester_aif.AifType(3))
    fitfunc = lambda C: manchester_tofts.solve_LLS(C, aif, 0)[0:2]
  elif author == "amsterdam":
    fitfunc = lambda C: amsterdam_wrapper(t, ca, C)
  else:
    print("UNKNOWN AUTHOR")
    return
  
  # Apply fitting function to data
  if not requires_1D:
    _kt, _ve = fitfunc(ct)
  else:
    _kt = np.zeros(numvox)
    _ve = np.zeros(numvox)
    for idx in range(numvox):
        _kt[idx], _ve[idx] = fitfunc(ct[idx, :])
  # Recreate the initial spatial dimensions of the input
  kt = _kt.reshape(spatialdims)
  ve = _ve.reshape(spatialdims)
  return (kt, ve)


def amsterdam_wrapper(t, ca, ct):
  # This implementation fits for kep and ve, so we need an extra step to return ktrans and ve
  AIF = amsterdam.fit_aif(ca, t, model='Cosine8')
  _kep, _dt, _ve, _vp = amsterdam.fit_tofts_model(ct, t, AIF, idxs=None, X0=(0.01, 0.01, 0.01, 0), bounds=((0.0, 0, 0.0, 0.0), (35, 1, 1, 1e-10)), jobs=1, model='Cosine8')
  _kt = _kep * _ve
  return (_kt, _ve)


def show_maps(fits = None, kt = None, ve = None, truth = None, returnfig = False):
    if not np.any(kt):
        kt, ve = fits

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
    ax1.imshow(kt)
    ax1.grid(False)
    plt.sca(ax1)
    plt.title("Ktrans")
    if np.any(truth):
      xticklocs, yticklocs = get_ticklocs(kt)
      plt.xticks(xticklocs, np.unique(truth.ve), rotation=45, ha="right")
      plt.yticks(yticklocs, np.unique(truth.kt))
      plt.xlabel("ve")
      plt.ylabel("kt")
    
    ax2.imshow(ve)
    ax2.grid(False)
    plt.sca(ax2)
    plt.title("ve")
    if np.any(truth):
      xticklocs, yticklocs = get_ticklocs(ve)
      plt.xticks(xticklocs, np.unique(truth.ve), rotation=45, ha="right")
      plt.yticks(yticklocs, np.unique(truth.kt))
      plt.xlabel("ve")
      plt.ylabel("kt")
   
    if returnfig:
        return fig
    return

def show_error_maps(fits, idx = -1, title = "", showcbar = False, truth = None, clim = (-100, 100), returnfig = False):
    kt, ve = fits

    if (idx >= 0):
        kt = kt[:,:,idx]
        ve = ve[:,:,idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
    ax1.imshow(percenterror(kt, truth.kt), cmap = "PuOr_r", clim = clim)
    ax1.grid(False)
    plt.sca(ax1)
    plt.title("%error in ktrans")
    if np.any(truth):
      xticklocs, yticklocs = get_ticklocs(kt)
      plt.xticks(xticklocs, np.unique(truth.ve), rotation=45, ha="right")
      plt.yticks(yticklocs, np.unique(truth.kt))
      plt.xlabel("ve")
      plt.ylabel("kt")
    
    im = ax2.imshow(percenterror(ve, truth.ve), cmap = "PuOr_r", clim = clim)
    ax2.grid(False)
    plt.sca(ax2)
    plt.title("%error in ve")
    if np.any(truth):
      xticklocs, yticklocs = get_ticklocs(ve)
      plt.xticks(xticklocs, np.unique(truth.ve), rotation=45, ha="right")
      plt.yticks(yticklocs, np.unique(truth.kt))
      plt.xlabel("ve")
      plt.ylabel("kt")
    
    fig.suptitle(title)
    if showcbar:
        fig.colorbar(im, ax=[ax1,ax2], orientation='vertical', shrink = 0.8)

    if returnfig:
        return fig
    return
  
def get_ticklocs(arr):
    if arr.shape[0] > 20:
        # Full resolution maps - 60x50
        xticklocs = np.arange(5, 50, 10)
        yticklocs = np.arange(5, 60, 10)
    else:
        # Downsampled maps - 6x5
        xticklocs = np.arange(0, 5, 1)
        yticklocs = np.arange(0, 6, 1)
    return xticklocs, yticklocs