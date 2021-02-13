"""
Model for the Detection device which can be either the Allied Vision
CCD camera or the Ando Spectrometer

An example of how to run the code is found at the end of this file.

File name: acquisition.py
Author: Julian Krauth
Date created: 2019/12/08
Python Version: 3.7
"""
import numpy as np

from labdevices import ando as ANDO
from labdevices import allied_vision


class Spectrometer:
    """This class defines the model for the Spectrometer."""
    mode = None # Is set upon initialization

    def __init__(self, test=True):
        if test:
            self.ando = ANDO.SpectrumAnalyzerDummy()
            self.camera = allied_vision.MantaDummy()
        else:
            self.ando = ANDO.SpectrumAnalyzer()
            self.camera = allied_vision.Manta(camera_id='DEV_000F314E1E59')

    def initialize(self, mode):
        """Connect to the device."""
        if mode == 0:
            self.camera.initialize()
        elif mode == 1:
            self.ando.initialize()
        self.mode = mode

    def get_ando_spectrum(self):
        """Get spectrum from Ando Spectrometer"""
        self.ando.sweep()
        self.ando.finish()
        y = self.ando.get_y_data()
        return y

    def get_camera_spectrum(self):
        """Get spectrum from ccd camera"""
        img = self.camera.take_single_img()
        # Project image onto a single axis and normalize
        y = np.divide(np.sum(img,0), float(np.ma.size(img,0)))
        return y

    def get_spectrum(self):
        """Get spectrum from the connected device (ando or ccd)"""
        if self.mode == 0:
            return self.get_camera_spectrum()
        if self.mode == 1:
            return self.get_ando_spectrum()

    def close(self):
        """Close connection to device."""
        if self.mode == 0:
            self.camera.close()
        elif self.mode == 1:
            self.ando.close()

    def exposure(self, val=None):
        """Set exposure of the camera"""
        if val is None:
            return self.camera.exposure
        else:
            self.camera.exposure = val

    def gain(self, val=None):
        """Set gain of the camera"""
        if val is None:
            return self.camera.gain
        else:
            self.camera.gain = val

    def img_format(self, offsetx=None, offsety=None,
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
            if offsetx is not None:
                self.camera.roi_x = offsetx
            if offsety is not None:
                self.camera.roi_y = offsety
            if width is not None:
                self.camera.roi_dx = width
            if height is not None:
                self.camera.roi_dy = height

    def trig_source(self, source=None):
        if source is None:
            return self.camera.trig_source()
        else:
            self.camera.trig_source(source)

    def pix_format(self, pix=None):
        if pix is None:
            return self.camera.pix_format()
        else:
            self.camera.pix_format(pix)

    def img_format_full(self):
        """Set image format to full size of camera sensor"""
        self.camera.roi_x = 0
        self.camera.roi_y = 0
        self.camera.roi_dx = self.camera.sensor_size[0]
        self.camera.roi_dy = self.camera.sensor_size[1]

    def take_single_img(self):
        return self.camera.take_single_img()

    def take_full_img(self):
        """Saves current roi parameters, changes to full sensor size,
        takes full image, restores old roi parameters in the settings."""
        x = self.camera.roi_x
        y = self.camera.roi_y
        dx = self.camera.roi_dx
        dy = self.camera.roi_dy
        self.img_format_full()
        img = self.take_single_img()
        self.img_format(x, y, dx, dy)
        return img


    def ctr(self, wl=None):
        if wl is None:
            return self.ando.ctr()
        else:
            self.ando.ctr(wl)

    def span(self, span=None):
        if span is None:
            return self.ando.span()
        else:
            self.ando.span(span)

    def cw_mode(self, cw=None):
        if cw is None:
            return self.ando.cw_mode()
        else:
            self.ando.cw_mode(cw)

    def peak_hold_mode(self, time):
        self.ando.peak_hold_mode(time)
