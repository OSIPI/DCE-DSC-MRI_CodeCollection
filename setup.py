from setuptools import setup, find_packages


setup(
        name='src',
        version = '1.0.0',
        install_requires = ['pytest','numpy','scipy','lmfit','pandas',
                            'joblib','matplotlib','dicom==0.9.9','progressbar',
                            'openpyxl'],
        packages = find_packages()
        )
