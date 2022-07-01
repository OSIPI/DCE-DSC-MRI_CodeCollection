from setuptools import setup, find_packages

setup(
    name='osipi_dce_dsc_repo',
    version='0.0.2',
    install_requires=['dicom==0.9.9.post1', 'imageio==2.19.3', 'joblib==1.1.0', 'lmfit==1.0.3',
                      'mat73==0.59', 'matplotlib==3.5.2', 'nibabel==3.2.2', 'numpy==1.23.0rc2',
                      'opencv-python==4.5.5.64', 'openpyxl==3.2.0b1', 'pandas==1.4.2',
                      'progressbar==2.5', 'pytest==7.1.2', 'scipy==1.8.1'],
    include_package_data=True,
    package_dir={'osipi_dce_dsc_repo': 'src'},
    packages=[f'osipi_dce_dsc_repo.{module}' for module in find_packages('src')],
)

