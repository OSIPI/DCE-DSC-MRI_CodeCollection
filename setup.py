from setuptools import setup, find_packages

setup(
    name='osipi_code_collection',
    version='0.0.2',
    install_requires=['dicom', 'imageio', 'joblib', 'lmfit',
                      'mat73', 'matplotlib', 'numpy', 'opencv-python', 'openpyxl',
                      'pandas', 'progressbar', 'pytest', 'scipy'],
    include_package_data=True,
    package_dir={'osipi_code_collection': 'src'},
    packages=[f'osipi_code_collection.{module}' for module in find_packages('src')],
)

