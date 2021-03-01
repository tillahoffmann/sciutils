import re
from setuptools import setup, find_packages

# Load the long description
with open('README.rst') as fp:
    long_description = fp.read()
long_description = long_description.replace('.. doctest::', '.. code-block:: python')
long_description = re.sub(r'(\.\. automodule:: .*?$)|(\.\. toctree::)', r':code:`\1`',
                          long_description, flags=re.MULTILINE)

# Load the version number
try:
    with open('VERSION') as fp:
        version = fp.read().strip()
except FileNotFoundError:
    version = 'dev'

setup(
    name='sciutils',
    version=version,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
    ],
    extras_require={
        'tests': [
            'flake8',
            'pytest',
            'pytest-cov',
        ],
        'docs': [
            'sphinx',
        ],
    },
    long_description_content_type='text/x-rst',
    long_description=long_description,
)
