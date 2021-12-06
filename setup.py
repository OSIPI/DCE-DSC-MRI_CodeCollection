from setuptools import setup, find_packages


setup(
        name='osipi',
        version = '0.0.1', 
        install_requires = ['imageio', 'joblib', 'lmfit', 'mat73', 'matplotlib', 'numpy', 'pandas', 'pytest', 'scipy', 'dicom', 'progressbar', 'cv2', 'opencv-python'],
        package_dir={'osipi': 'src'},
        packages=[f'osipi.{module}' for module in find_packages('src')],
        )
        