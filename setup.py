"""
Installation file for the shg_frog package.

File name: setup.py
Author: Julian Krauth
Date created: 2021/02/16
"""
import pathlib
from setuptools import setup, find_packages
from shg_frog import __version__

HERE = pathlib.Path(__file__).parent

README = (HERE / 'README.md').read_text()

with open(HERE / 'requirements.txt') as fh:
    requirements = [line.strip() for line in fh]

setup(
    name='shg_frog',
    version=__version__,
    description='Software for frequency-resolved optical gating measurements of ultra-fast laser pulses.',
    long_description=README,
    url='https://github.com/jkrauth/shg_frog',
    author='Julian Krauth',
    author_email='j.krauth@vu.nl',
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
    ],
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'shg_frog = shg_frog.__main__:main',
        ],
    }
)
