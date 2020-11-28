import os
import re
from setuptools import setup, find_packages

# Load the long description
with open('README.rst') as fp:
    long_description = fp.read()
long_description = long_description.replace('.. doctest::', '.. code-block:: python')
long_description = re.sub(r'(\.\. automodule:: .*?$)', r':code:`\1`', long_description,
                          flags=re.MULTILINE)

# Automatically determine the version to push to pypi
github_ref = os.environ.get('GITHUB_REF', '')
prefix = 'refs/tags/v'
if github_ref.startswith(prefix):
    version = github_ref[len(prefix):]
else:
    version = 'dev'

setup(
    name='sciutils',
    version=version,
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    extras_require={
        'tests': [
            'flake8',
            'pytest',
            'pytest-cov',
        ],
        'docs': [
            'sphinx',
        ]
    },
    long_description_content_type='text/x-rst',
    long_description=long_description,
)
