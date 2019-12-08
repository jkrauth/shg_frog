"""
Model for the Detection device which can be either the Allied Vision
camera or the Ando Spectrometer

An example of how to run the code is found at the end of this file.

File name: acquisition.py
Author: Julian Krauth
Date created: 2019/12/08
Python Version: 3.7
"""

import os
import sys
import numpy as np

cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_dir)
python_dir = os.path.join(cur_dir, "..", "..", "Python")
sys.path.append(python_dir)
print(python_dir)
from Controller import ANDO_SPECTROMETER, ALLIED_VISION_CCD

class Spectrometer:

    def __init__(self, test=True):
        if test:
            self.ando = ANDO_SPECTROMETER.AndoSpectrumAnalyzerDummy()
            self.camera = ALLIED_VISION_CCD.CCDcameraDummy()
        else:
            self.ando = ANDO_SPECTROMETER.AndoSpectrumAnalyzer()
            self.camera = ALLIED_VISION_CCD.CCDcamera()

    def initialize(self, mode):
        if mode == 0:
            self.camera.initialize()
        elif mode == 1:
            self.ando.initialize()

    def get_ando_spectrum(self):
        self.ando.sweep()
        self.ando.finish()
        y = self.ando.get_y_data()
        return y

    def get_camera_spectrum(self):
        img = self.camera.takeSingleImg()
        # Get maximum pixel value
        if max_of_trace < np.amax(img):
            max_of_trace = np.amax(img)
        # Project image onto a single axis and normalize
        y = np.divide(np.sum(img,0),float(np.ma.size(img,0)))
        return y

    def get_spectrum(self, mode):
        if mode == 0:
            return self.get_camera_spectrum()
        if mode == 1:
            return self.get_ando_spectrum()

    def close(self, mode):
        if mode == 0:
            self.camera.close()
        elif mode == 1:
            self.ando.close()
