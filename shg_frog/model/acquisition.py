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

from labdevices import ando, allied_vision


class _CameraMixin:
    """ Extension/Mixin for the Manta camera class """
    def get_spectrum(self) -> np.ndarray:
        """Get spectrum from ccd camera"""
        img = self.take_single_img()
        # Project image onto a single axis and normalize
        y_data = np.divide(np.sum(img, 0), float(np.ma.size(img, 0)))
        return y_data

    def get_roi(self) -> list:
        """ Gives the region of interest. """
        img_format = [0]*4
        img_format[0] = self.roi_x
        img_format[1] = self.roi_y
        img_format[2] = self.roi_dx
        img_format[3] = self.roi_dy
        return img_format

    def set_roi(self, offsetx=None, offsety=None,
            width=None, height=None) -> None:
        """
        Set region of interest of the image which is acquired from the camera chip.
        (It can be just a fraction of the full format)
        Units in pixels!
        """
        if offsetx is not None:
            self.roi_x = offsetx
        if offsety is not None:
            self.roi_y = offsety
        if width is not None:
            self.roi_dx = width
        if height is not None:
            self.roi_dy = height

    def img_format_full(self) -> None:
        """Set image format to full size of camera sensor"""
        self.roi_x = 0
        self.roi_y = 0
        self.roi_dx = self.sensor_size[0]
        self.roi_dy = self.sensor_size[1]

    def take_full_img(self) -> np.ndarray:
        """Saves current roi parameters, changes to full sensor size,
        takes full image, restores old roi parameters in the settings."""
        x_old = self.roi_x
        y_old = self.roi_y
        dx_old = self.roi_dx
        dy_old = self.roi_dy
        self.img_format_full()
        image = self.take_single_img()
        self.set_roi(x_old, y_old, dx_old, dy_old)
        return image

    # Define callables needed for pyqt connect functions
    def set_exposure(self, exposure):
        self.exposure = exposure

    def set_gain(self, gain):
        self.gain = gain

    def set_trig_source(self, source):
        self.trig_source = source

class Camera(allied_vision.Manta, _CameraMixin):
    """ Manta camera with some additional features. """

class CameraDummy(allied_vision.MantaDummy, _CameraMixin):
    """ Manta camera dummy with some additional features """

class _SpectrometerMixin:
    """Extension/Mixin for the Spectrum analyzer."""

    def get_spectrum(self):
        """Get spectrum from Ando Spectrometer"""
        self.do_sweep()
        self.finish()
        y_data = self.get_y_data()
        return y_data

class Spectrometer(ando.SpectrumAnalyzer, _SpectrometerMixin):
    """ Spectrometer with additional features. """

class SpectrometerDummy(ando.SpectrumAnalyzerDummy, _SpectrometerMixin):
    """ Spectrometer Dummy with additional features. """
