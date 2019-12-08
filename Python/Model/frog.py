"""
Model for the FROG setup

An example of how to run the code is found at the end of this file.

File name: frog.py
Author: Julian Krauth
Date created: 2019/12/02
Python Version: 3.7                     
"""

import os
import sys

cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_dir)
python_dir = os.path.join(cur_dir, "..", "..", "Python")
sys.path.append(python_dir)
print(python_dir)
from Controller import NEWPORT
from Model import acquisition

class FROG:

    def __init__(self, test=True):
        if test:
            self.stage = NEWPORT.SMC100DUMMY(port='/dev/ttyUSB0', dev_number=1)
        else:
            self.stage = NEWPORT.SMC100(port='/dev/ttyUSB0', dev_number=1)
        self.spect = acquisition.Spectrometer(test)

    def initialize(self, mode=0):
        self.stage.initialize()
        self.spect.initialize(mode)

    def set_path_diff(self):
        pass

    

    def close(self, mode=0):
        self.stage.close()
        self.spect.close(mode)



if __name__ == "__main__":

    frog = FROG()
    frog.initialize()
    frog.close()