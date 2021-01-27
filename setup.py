from setuptools import setup, find_packages
from codecs import open

import deep_gas_oracle


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='deep_gas_oracle',
    version=deep_gas_oracle.version,
    description='Deep Gas Oracle for Ethereum, using Recurrent Neural Networks.',
    url='https://github.com/louisoutin',
    author='Louis Outin',
    author_email='louis.outin@gmail.com',
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=requirements,
)
