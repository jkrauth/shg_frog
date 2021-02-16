"""
Installation file for the shg_frog package.

File name: setup.py
Author: Julian Krauth
Date created: 2021/02/16
"""
from setuptools import setup, find_packages

setup(
    name='shg_frog',
    version=0.1,
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'pyqt5',
        'pyqtgraph',
        'matplotlib',
        'pyyaml',
        'imageio',
    ],
    entry_points={
        'console_scripts': [
            'shg_frog = shg_frog.start_frog:main',
        ],
    }
)
