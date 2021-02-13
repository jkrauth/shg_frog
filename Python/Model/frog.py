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
from time import sleep
from datetime import datetime
import numpy as np
import yaml

from labdevices import newport


cur_dir = os.path.abspath(os.path.dirname(__file__))
python_dir = os.path.join(cur_dir, "..", "..", "Python")
sys.path.append(python_dir)
config_dir = os.path.join(python_dir, "..", "Config")

from Model import acquisition
from Model import phase_retrieval

SPEEDOFLIGHT = 299792458. #m/s

class FROG:
    """Top level class for the FROG experiment definition."""
    # Get defaults from config file
    with open(config_dir + '/config.yml','r') as f:
        CONFIG = yaml.load(f,Loader=yaml.FullLoader)
    DEFAULTS = {
        'center wavelength': CONFIG['center wavelength'],
        'focal length': CONFIG['focal length'],
        'grating': CONFIG['grating'],
        'pixel size': CONFIG['pixel size'],
    }

    def __init__(self, test=True):
        # Load the FROG devices (optional: virtual devices for testing)
        if test:
            self.stage = newport.SMC100DUMMY(port='/dev/ttyUSB0', dev_number=1)
        else:
            self.stage = newport.SMC100(port='/dev/ttyUSB0', dev_number=1)
        self.spect = acquisition.Spectrometer(test)

        self.measured_trace = None
        self.used_settings = None
        self.stop_measure = False

        self.algo = phase_retrieval.PhaseRetrieval()

    def initialize(self, mode=0):
        """Connect to the devices."""
        self.stage.initialize()
        self.spect.initialize(mode)


    def measure(self, sig_progress, sig_measure, start_pos, max_meas, step_size):
        """Carries out the measurement loop."""
        # Delete previously measured trace from memory.
        self.measured_trace = None
        self.used_settings = None
        # Move stage to Start Position and wait for end of movement
        self.stage.goto(start_pos)
        self.stage.wait_move_finish(1.)
        for i in range(max_meas):
            print("Loop...")
            # Move stage
            self.stage.goto(start_pos+i*step_size)
            self.stage.wait_move_finish(0.2)
            # Record spectrum
            y = self.spect.get_spectrum()
            # Create 2d frog-array to fill with data
            if i==0:
                frog_array = np.zeros((len(y),max_meas))
            # Stitch data together
            frog_array[:,i] = y
            # Send data to plot
            sig_measure.emit(3, y)
            sig_measure.emit(2, frog_array)
            sleep(0.2)
            sig_progress.emit(i+1)
            if self.stop_measure:
                break
        if self.stop_measure:
            print("Measurement aborted!")
        else:
            # Save Frog trace and measurement settings as instance attributes,
            # they are then available for save button of GUI.
            self.measured_trace = self.scale_pxl_values(frog_array)
            self.used_settings = self.save_settings(step_size, max_meas)
            print("Measurement finished!")

    def scale_pxl_values(self, frog_array):
        """Maximize contrast of the image"""
        if self.spect.mode == 0: # for ccd/cmos camera
            # Scale image according to bit depth
            pix_format = self.spect.camera.pix_format()
            if pix_format == 'Mono8':
                scale = 255.
            elif pix_format == 'Mono12':
                scale = 65535.
            frog_array_scaled = np.rint(scale * frog_array / np.amax(frog_array)).astype(int)
        elif self.spect.mode == 1: # for ANDO
            raise Exception("scaling for ando not implemented yet.")
            # Maybe there is no scaling needed...
        return frog_array_scaled

    def save_settings(self, step_size, step_num: int):
        """Returns the settings of the last measurement as dictionary"""
        date = datetime.now().strftime('%Y-%m-%d')
        time = datetime.now().strftime('%H:%M:%S')
        # Time step per pixel in ps
        ccddt = 1e6*2*step_size/(SPEEDOFLIGHT)
        ccddv = self.freq_step_per_pixel()
        # in future maybe write also exposure time, gain, max Intensity, bit depth
        settings = {
            'date': date,
            'time': time,
            'center position': None,
            'step size': step_size,
            'step number': step_num,
            'ccddt': ccddt,
            'ccddv': ccddv,

        }
        return settings

    def freq_step_per_pixel(self):
        """Returns the frequency step per bin/pixel of the taken trace.
        Needed for phase retrieval.
        """
        if self.spect.mode == 0: # for CCD camera
            wlatcenter = self.DEFAULTS['center wavelength']
            # Wavelength step per pixel:
            # I assume that over the size of the CCD chip
            # (for small angles) the wavelength scale is linear
            # The number is calculated using the wavelength spread per mrad
            # specified for the grating.
            # This is then divided by the number of pixels which fit
            # into a 1mrad range at the focal distance of the lens:
            # Grating spec: 0.81nm/mrad => 0.81nm/0.2mm (for a 200mm focal lens)
            # =>0.81nm/34.13pixels (for 5.86micron pixelsize)
            mm_per_mrad = 1. * self.DEFAULTS['focal length'] / 1000.
            pxls_per_mrad = mm_per_mrad/(self.DEFAULTS['pixel size'] \
                /1000) # yields 34
            nm_per_px = self.DEFAULTS['grating']/pxls_per_mrad # yields 0.0237nm
            # Frequency step per pixel
            vperpxGHz = SPEEDOFLIGHT * (1/(wlatcenter) \
                -1/(wlatcenter + nm_per_px)) # GHz
            vperpx = vperpxGHz * 1.e-3 # THz
            # Also here I assume that for small angles the frequency can be
            # considered to be linear on the CCD plane.

        elif self.spect.mode == 1: # for ANDO spectrometer
            # One has to get that information from the ando settings.
            raise Exception("Calibration for ANDO spectrometer not yet implemented!")
        return vperpx

    def retrieve_phase(
        self, sig_retdata, sig_retlabels, sig_rettitles, sig_retaxis,
        pixels, GTol, iterMAX):
        """Execute phase retrieval algorithm."""
        if self.measured_trace is not None:
            ccddt = self.used_settings['ccddt']
            ccddv = self.used_settings['ccddv']
            self.algo.prepFROG(ccddt=ccddt, ccddv=ccddv, N=pixels, \
                ccdimg=self.measured_trace, flip=2)
            self.algo.retrievePhase(GTol=GTol, iterMAX=iterMAX, signal_data=sig_retdata, \
                signal_label=sig_retlabels, signal_title=sig_rettitles, signal_axis=sig_retaxis)
        else:
            raise Exception('No recorded trace in buffer!')

    def close(self):
        """Close connection with devices."""
        self.stage.close()
        self.spect.close()



if __name__ == "__main__":

    frog = FROG()
    frog.initialize()
    frog.close()
