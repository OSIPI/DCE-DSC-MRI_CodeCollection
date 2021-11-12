import urllib.request
import os
import mat73
import scipy.io

def add(x, y):
    return x + y

def remove_spaces(data):
    return data.replace(' ', '')

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
  return mat
    
def percenterror(estimated, truth):
    return 100 * (estimated - truth) / truth