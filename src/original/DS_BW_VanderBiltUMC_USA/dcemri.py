# Pyhton module for DCE-MRI postprocessing
#
# Copyright (C) 2014   David S. Smith
#

import time
from pylab import *
from scipy.integrate import cumtrapz, simps
from scipy.optimize import curve_fit


def status_check(k, N, tstart, nupdates=10):
    increment = int(N / nupdates)
    if (k + 1) % increment == 0:
        pct_complete = 100.0 * float(k + 1) / float(N)
        telapsed = time.time() - tstart
        ttotal = telapsed * 100.0 / pct_complete
        trem = ttotal - telapsed
        print("%.0f%% complete, %d of %d s remain".format(pct_complete, trem, ttotal))
    if k == N - 1:
        print("%d s elapsed".format(time.time() - tstart))


def signal_to_noise_ratio(im1, im2, mask=None, thresh=None):
    """Compute SNR of two images (see Dietrich et al. 2007,
    JMRI, 26, 375)"""
    print("computing signal-to-noise ratio")
    from skimage.filter import threshold_otsu

    if mask is None:
        if thresh is None:
            thresh = threshold_otsu(im1)
        mask = im1 > thresh
    return (
        (im1[mask] + im2[mask]).mean() / (im1[mask] - im2[mask]).std() / sqrt(2),
        mask,
    )


def signal_enhancement_ratio(data, thresh=0.01):
    """Compute max signal enhancement ratio for dynamic data"""
    print("computing signal enhancement ratios")
    assert thresh > 0.0
    ndyn = data.shape[-1]
    image_shape = data.shape[:-1]
    SER = zeros(image_shape, dtype=data.dtype)
    data = reshape(data, (-1, ndyn))
    S0 = data[:, 0].flatten()
    mask_ser = S0 > thresh * data.max()
    SER = data.max(axis=1).flatten()
    SER[mask_ser] /= S0[mask_ser]
    SER[~mask_ser] = 0
    SER = reshape(SER, image_shape)
    return SER


def dce_to_r1eff(S, S0, R1, TR, flip):
    print("converting DCE signal to effective R1")
    assert flip > 0.0
    assert TR > 0.0 and TR < 1.0
    S = S.T
    S0 = S0.T
    A = S.copy() / S0  # normalize by pre-contrast signal
    E0 = exp(-R1 * TR)
    E = (1.0 - A + A * E0 - E0 * cos(flip)) / (
        1.0 - A * cos(flip) + A * E0 * cos(flip) - E0 * cos(flip)
    )
    R = (-1.0 / TR) * log(E)
    return R.T


def dce_to_r1eff_old(S, S0map, idxs, TR, flip):
    """Convert DCE signal to effective R1, based on the FLASH signal equation"""
    T = zeros_like(S)
    T[idxs, :] = (
        S[idxs, :].T / S0map.flat[idxs] / sin(flip)
    ).T  # normalize by pre-contrast signal
    R1 = zeros_like(T)
    R1[idxs, :] = -log((T[idxs, :] - 1) / (T[idxs, :] * cos(flip) - 1)) / TR
    return R1


def r1eff_to_conc(R1eff, R1map, relaxivity):
    print("converting effective R1 to tracer tissue concentration")
    assert relaxivity > 0.0
    return (R1eff - R1map) / relaxivity


def ext_tofts_integral(t, Cp, Kt=0.1, ve=0.2, vp=0.1, uniform_sampling=True):
    """Extended Tofts Model, with time t in min.
    Works when t_dce = t_aif only and t is uniformly spaced.
    """
    nt = len(t)
    Ct = zeros(nt)
    for k in range(nt):
        if uniform_sampling:
            tmp = (
                cumtrapz(
                    exp(-Kt * (t[k] - t[: k + 1]) / ve) * Cp[: k + 1],
                    t[: k + 1],
                    initial=0.0,
                )
                + vp * Cp[: k + 1]
            )
            Ct[k] = tmp[-1]
        else:
            Ct[k] = (
                simps(exp(-Kt * (t[k] - t[: k + 1]) / ve) * Cp[: k + 1], t[: k + 1])
                + vp * Cp[: k + 1]
            )
    return Ct * Kt


def tofts_integral(t, Cp, Kt=0.1, ve=0.2, uniform_sampling=True):
    """Standard Tofts Model, with time t in min.
    Current works only when AIF and DCE data are sampled on
    same grid."""
    nt = len(t)
    Ct = zeros(nt)
    for k in range(nt):
        if uniform_sampling:
            tmp = cumtrapz(
                exp(-(Kt / ve) * (t[k] - t[: k + 1])) * Cp[: k + 1],
                t[: k + 1],
                initial=0.0,
            )
            Ct[k] = tmp[-1]
            # Ct[k] = simps(exp(-(Kt/ve)*(t[k] - t[:k+1]))*Cp[:k+1],
            #              dx=t[1]-t[0])
        else:
            Ct[k] = simps(
                exp(-(Kt / ve) * (t[k] - t[: k + 1])) * Cp[: k + 1], x=t[: k + 1]
            )
    return Kt * Ct


def fit_tofts_model(Ct, Cp, t, idxs=None, extended=False, plot_each_fit=False):
    """Solve tissue model for each voxel and return parameter maps.

    Ct: tissue concentration of CA, expected to be N x Ndyn

    t: time samples, assumed to be the same for Ct and Cp

    extended: if True, use Extended Tofts-Kety model.

    idxs: indices of ROI to fit
    """
    print("fitting perfusion parameters")
    N, ndyn = Ct.shape
    Kt = zeros(N)
    ve = zeros(N)
    Kt_cov = zeros(N)
    ve_cov = zeros(N)

    if idxs is None:
        idxs = range(N)

    # choose model and initialize fit parameters with reasonable values
    if extended:  # add vp if using Extended Tofts
        print("using Extended Tofts-Kety")
        fit_func = lambda t, Kt, ve, vp: ext_tofts_integral(t, Cp, Kt=Kt, ve=ve, vp=vp)
        coef0 = [0.01, 0.01, 0.01]
        popt_default = [-1, -1, -1]
        pcov_default = ones((3, 3))
    else:
        print("using Standard Tofts-Kety")
        vp = zeros(N)
        vp_cov = zeros(N)
        fit_func = lambda t, Kt, ve: tofts_integral(t, Cp, Kt=Kt, ve=ve)
        coef0 = [0.01, 0.01]
        popt_default = [-1, -1]
        pcov_default = ones((2, 2))

    print("fitting %d voxels" % len(idxs))
    tstart = time.time()
    for k, idx in enumerate(idxs):
        try:
            popt, pcov = curve_fit(fit_func, t, Ct[idx, :], p0=coef0)
        except RuntimeError:
            popt = popt_default
            pcov = pcov_default
        Kt[idx] = popt[0]
        ve[idx] = popt[1]
        try:
            Kt_cov[idx] = pcov[0, 0]
            ve_cov[idx] = pcov[1, 1]
        except TypeError:
            None  # print idx, popt, pcov
        if extended:
            vp[idx] = popt[2]
            vp_cov[idx] = pcov[2, 2]
        if plot_each_fit:
            figure(1)
            clf()
            plot(t, Ct[idx, :], "bo", alpha=0.6)
            plot(t, fit_func(t, *popt), "m-")
            pause(1)
            show()
        status_check(k, len(idxs), tstart=tstart)

    # bundle parameters for return
    params = [Kt, ve]
    stds = [sqrt(Kt_cov), sqrt(ve_cov)]
    if extended:
        params.append(vp)
        stds.append(sqrt(vp_cov))
    return (params, stds)


def fit_R1(images, flip_angles, TR):
    """Create T1 map from multiflip images"""
    inshape = images.shape
    nangles = inshape[-1]
    n = prod(inshape[:-1])
    images = reshape(images, (n, nangles))
    # flip_angles = pi*arange(20,0,-2)/180.0  # deg
    assert nangles == len(flip_angles)
    signal_scale = abs(images).max()
    images = images / signal_scale
    R1map = zeros(n)
    S0map = zeros(n)
    covmap = zeros((n, 4))

    def t1_signal_eqn(x, M0, R1):
        E1 = exp(-TR * R1)
        return M0 * sin(x) * (1.0 - E1) / (1.0 - E1 * cos(x))

    # fit_func = lambda x, y, z: t1_signal_eqn(x, y, z, TR)
    for j in range(n):
        if images[j, :].mean() > 0.1:
            try:
                popt, pcov = curve_fit(t1_signal_eqn, flip_angles, images[j, :].copy())
            except RuntimeError:
                popt = [0, 0]
                pcov = array([[0, 0], [0, 0]])
            S0map[j] = popt[0]
            R1map[j] = popt[1]
            if not isinstance(pcov, float):
                covmap[j, :] = pcov.flatten()
    S0map = S0map * signal_scale
    images = images * signal_scale
    images = reshape(images, inshape)
    return (R1map, S0map, covmap)


def process(
    dcefile, t1file, t1_flip, R, TE, TR, dce_flip, extended=False, plotting=False
):
    """Compute perfusion parameters for a DCE-MRI data set."""

    return None
