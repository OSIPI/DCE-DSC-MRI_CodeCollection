import pytest
import numpy as np
from src.original.DS_BW_VanderBiltUMC_USA.dcemri import tofts_integral, ext_tofts_integral, signal_to_noise_ratio

def test_tofts_integral_basics():
    # Define time points (0 to 10 minutes)
    t = np.linspace(0, 10, 20)
    # Define an idealized AIF (exponential decay)
    Cp = np.exp(-t)
    
    # Run the model
    Kt = 0.2
    ve = 0.1
    Ct = tofts_integral(t, Cp, Kt=Kt, ve=ve)
    
    # Checks
    assert Ct.shape == t.shape
    assert np.all(np.isfinite(Ct))  # No NaNs or Infs
    assert np.max(Ct) > 0  # Should have some signal

def test_extended_tofts_integral_basics():
    t = np.linspace(0, 10, 20)
    Cp = np.exp(-t)
    
    # Run extended model (adds vp)
    Kt = 0.2
    ve = 0.1
    vp = 0.05
    Ct = ext_tofts_integral(t, Cp, Kt=Kt, ve=ve, vp=vp)
    
    assert Ct.shape == t.shape
    assert np.all(np.isfinite(Ct))

def test_signal_to_noise_ratio_basics():
    # Create two identical images with slight noise difference
    img1 = np.ones((10, 10))
    img2 = np.ones((10, 10)) + 0.1 * np.random.rand(10, 10)
    
    # This function prints to stdout, so we just check it runs without crashing
    snr, mask = signal_to_noise_ratio(img1, img2)
    
    assert isinstance(snr, float)
    assert mask.shape == img1.shape