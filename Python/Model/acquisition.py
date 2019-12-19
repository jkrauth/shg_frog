"""
Model for the Detection device which can be either the Allied Vision
CCD camera or the Ando Spectrometer

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

    def exposure(self, val=None):
        if val is None:
            return self.camera.exposure
        else:
            self.camera.exposure = val

    def gain(self, val=None):
        if val is None:
            return self.camera.gain
        else:
            self.camera.gain = val
            

    def imgFormat(self, offsetx=None, offsety=None,
                  width=None, height=None):
        """
        Get/Set position and format of the image which is acquired
        from the camera chip. (It can be just a fraction of the full
        format)
        Full format is [0, 0, 1936, 1216]
        Units in pixels!
        """
        if offsetx==offsety==width==height==None:
           img_format = np.zeros(4,dtype=int)
           img_format[0] = self.camera.roi_x
           img_format[1] = self.camera.roi_y
           img_format[2] = self.camera.roi_dx
           img_format[3] = self.camera.roi_dy
           return img_format
        else:
            if offsetx!=None:
                self.camera.roi_x = offsetx
            if offsety!=None:
                self.camera.roi_y = offsety
            if width!=None:
                self.camera.roi_dx = width
            if height!=None:
                self.camera.roi_dy = height

    def trigSource(self, source=None):
        if source is None:
            return self.camera.trigSource()
        else:
            self.camera.trigSource(source)

    def pixFormat(self, pix=None):
        if pix is None:
            return self.camera.pixFormat()
        else:
            self.camera.pixFormat(pix)
