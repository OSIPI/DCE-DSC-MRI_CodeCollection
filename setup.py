from setuptools import setup, find_packages


setup(
        name='src',
        version = '1.0.0', 
        install_requires = ['pytest','numpy','scipy','lmfit','pandas','joblib'],
        packages = find_packages()
        )
