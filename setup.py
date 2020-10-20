from setuptools import setup, find_packages


setup(
        name='src',
        version = '1.0.0', 
        install_requires = ['pytest','numpy','scipy'],
        packages = find_packages()
        )
