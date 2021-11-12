from setuptools import setup, find_packages


setup(
        name='osipi',
        version = '1.0.0', 
        install_requires = ['joblib', 'lmfit', 'mat73', 'matplotlib', 'numpy', 'pandas', 'pytest', 'scipy'],
        package_dir={'osipi': 'src'},
        packages=[f'osipi.{module}' for module in find_packages('src')],
        )
        