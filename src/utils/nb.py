import imageio
import mat73
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
from types import SimpleNamespace
import urllib.request

def download_osf(url, file, overwrite = False):
  if os.path.isfile(file):
    print("Destination file already exists")
    if overwrite:
      print(f"  Deleting: {file}")
      os.remove(file)
    else:
      print(f"  Skipping: {file}")
      return
  urllib.request.urlretrieve(url, file)
  return
    
def loadmat(matfile):
  try:
    mat = scipy.io.loadmat(matfile)
  except:
    mat = mat73.loadmat(matfile)
  return SimpleNamespace(**mat)
    
def percenterror(estimated, truth):
    return 100 * (estimated - truth) / truth

def make_error_map_gif(drawfunc, filename, title = ""):
  figs = []
  for idx in [0,1,2,3,4,5]:
      figs.append(drawfunc(idx = idx, title = f"{title} - {idx}"))
  figs2gif(figs, filename)
  plt.close('all')

def fig2image(fig):
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

def figs2gif(figs, gifname, fps = 1):
    imageio.mimsave(gifname, [fig2image(fig) for fig in figs], fps = fps)
    
