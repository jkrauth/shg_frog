"""
Model for the FROG setup

An example of how to run the code is found at the end of this file.

File name: frog.py
Author: Julian Krauth
Date created: 2019/12/02
Python Version: 3.7
"""
import pathlib
from time import sleep
from datetime import datetime
import numpy as np
import yaml
from pyqtgraph.parametertree import Parameter

from labdevices import newport

from . import acquisition
from . import phase_retrieval

MAIN_DIR = pathlib.Path(__file__).parents[2]
CONFIG_DIR = MAIN_DIR / "Config" / 'config.yml'
DATA_DIR = MAIN_DIR / "Data"

SPEEDOFLIGHT = 299792458. #m/s

class FROG:
    """Top level class for the FROG experiment definition."""

    def __init__(self, test: bool=True):
        """
        Arguments:
        test -- whether to run with dummy devices for testing or not.
        """
        # Load the FROG devices (optional: virtual devices for testing)
        if test:
            self.stage = newport.SMC100DUMMY(port='/dev/ttyUSB0', dev_number=1)
        else:
            self.stage = newport.SMC100(port='/dev/ttyUSB0', dev_number=1)
        self.spect = acquisition.Spectrometer(test)

        self._config = self._get_config()

        self.measured_trace = None
        self.used_settings = None
        self.stop_measure = False

        self.algo = phase_retrieval.PhaseRetrieval()
        self.parameters = FrogParams(self._config['pxls width'], self._config['pxls height'])

    def initialize(self, mode: int=0) -> None:
        """Connect to the devices."""
        self.stage.initialize()
        self.spect.initialize(mode)

    def _get_config(self) -> dict:
        """Get defaults from configuration file."""
        with open(CONFIG_DIR, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def get_data_path(self) -> pathlib.Path:
        return DATA_DIR

    def get_file_name(self) -> str:
        return self._config['file name']

    def get_image_suffix(self) -> str:
        return self._config['image suffix']

    def get_meta_suffix(self) -> str:
        return self._config['meta suffix']


    def measure(self, sig_progress, sig_measure, \
        start_pos: float, max_meas: int, step_size: float):
        """Carries out the measurement loop."""

        # Delete previously measured trace.
        self.measured_trace = None
        self.used_settings = None
        # Move stage to Start Position and wait for end of movement
        self.stage.move_abs(start_pos)
        self.stage.wait_move_finish(1.)
        for i in range(max_meas):
            print("Loop...")
            # Move stage
            self.stage.move_abs(start_pos+i*step_size)
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
            self.used_settings = self.get_settings(step_size, max_meas)
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

    def get_settings(self, step_size, step_num: int):
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
            'center position': self.parameters.par.param('Newport Stage').child('Offset').value(),
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
            wlatcenter = self._config['center wavelength']
            # Wavelength step per pixel:
            # I assume that over the size of the CCD chip
            # (for small angles) the wavelength scale is linear
            # The number is calculated using the wavelength spread per mrad
            # specified for the grating.
            # This is then divided by the number of pixels which fit
            # into a 1mrad range at the focal distance of the lens:
            # Grating spec: 0.81nm/mrad => 0.81nm/0.2mm (for a 200mm focal lens)
            # =>0.81nm/34.13pixels (for 5.86micron pixelsize)
            mm_per_mrad = 1. * self._config['focal length'] / 1000.
            pxls_per_mrad = mm_per_mrad/(self._config['pixel size'] \
                /1000) # yields 34
            nm_per_px = self._config['grating']/pxls_per_mrad # yields 0.0237nm
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



class FrogParams:
    """
    Class which implements the parameters used by the parametertree widget.
    """

    def __init__(self, sensor_width: int, sensor_height: int):
        """
        Arguments:
        sensor_width -- pixels along horizontal
        sensor_height -- pixels along vertical
        """
        # Define parameters for parametertree
        self._sensor_width = sensor_width
        self._sensor_height = sensor_height
        params = [
            {'name':'Phase Retrieval', 'type':'group', 'visible':True, 'children': [
                {'name':'prepFROG Size', 'type': 'int', 'value': 128},
                {'name':'Seed', 'type': 'list', 'values': {"Random":0,"fromFile":1}, 'value':0},
                {'name':'Max. Iterations', 'type': 'int', 'value': 200},
                {'name':'G Tolerance', 'type': 'float', 'value': 0.001}
            ]},
            {'name':'ANDO Spectrometer', 'type':'group', 'visible':False, 'children': [
                {'name':'Center', 'type': 'float', 'value': 390.},
                {'name':'Span', 'type': 'float', 'value': 20.},
                {'name':'CW mode', 'type': 'bool', 'value': True},
                {'name':'Rep. time', 'type': 'float', 'value': 36},
                {'name':'Sampling', 'type': 'int', 'value': 201}
            ]},
            {'name':'ALLIED VISION CCD', 'type':'group', 'visible':False, 'children': [
                {'name':'Exposure', 'type': 'float', 'value': 0.036, 'dec': True, \
                    'step': 1, 'siPrefix': True, 'suffix': 's'},
                {'name':'Gain', 'type': 'float', 'value': 0, 'dec': False, 'step': 1},
                {'name':'Crop Image', 'type': 'group', 'expanded':False, 'children': [
                    {'name':'Width','type':'int', \
                        'value': 600,'limits':[1, sensor_width],'step': 2},
                    {'name':'Height','type':'int', \
                        'value': 10,'limits':[1, sensor_height],'step': 2},
                    {'name':'Xpos','type':'int','value': 400,'step': 2},
                    {'name':'Ypos','type':'int','value': 470,'step': 2}
                ]},
                {'name':'Trigger', 'type': 'group', 'children': [
                    {'name':'Mode','type':'list','visible':False, \
                        'values': {"On":1,"Off":0},'value':1},
                    {'name':'Source','type':'list','visible':True, \
                        'values': {
                            "Freerun":'Freerun',
                            "External":'Line1'
                            #'Line2':'Line2',
                            #'FixedRate':'FixedRate',
                            #'Software':'Software'
                        },'value':'External'}
                ]},
                {'name':'PixelFormat', 'type': 'list', \
                    'values': {
                        'Mono8':'Mono8',
                        'Mono12':'Mono12'
                        #'Mono12Packed':'Mono12Packed'
                    }, 'value':'Mono8'},
                ]},
            {'name':'Newport Stage', 'type':'group', 'visible':False, 'children': [
                {'name':'Position', 'type':'float', 'value': 0., 'readonly': True},
                {'name':'GoTo', 'type':'float', 'value': 0.},
                {'name':'Offset', 'type':'float', 'value': 11370, 'limits':[0,25000]},
                {'name':'Start Position', 'type':'float', 'value': -256},
                {'name':'Step Size', 'type':'float', 'value': 4.},
                {'name':'Number of steps', 'type':'int', 'readonly':True, 'value': 128}
            ]}
        ]

        # Create parameter objects
        self.par = Parameter.create(name='params',
                              type='group',
                              children=params)

        ### Some settings regarding CCD parameters ###
        # Create limits for crop settings

        crop_par = self.par.param('ALLIED VISION CCD').child('Crop Image')
        width_par = crop_par.child('Width')
        height_par = crop_par.child('Height')
        xpos_par = crop_par.child('Xpos')
        ypos_par = crop_par.child('Ypos')
        width_par.setLimits([1, self._sensor_width-xpos_par.value()])
        height_par.setLimits([1, self._sensor_height-ypos_par.value()])
        xpos_par.setLimits([0, self._sensor_width-width_par.value()])
        ypos_par.setLimits([0, self._sensor_height-height_par.value()])
        crop_par.sigTreeStateChanged.connect(self.set_crop_limits)

        ### Some settings regarding the Stage parameters ###
        stage_par = self.par.param('Newport Stage')
        start_par = stage_par.child('Start Position')
        step_par = stage_par.child('Step Size')
        off_par = stage_par.child('Offset')

        # Set limits of Start Position, depending on offset
        start_par.setLimits([-off_par.value(),-0.2])
        off_par.sigValueChanged.connect(self.set_start_pos_limits)

        # Set limits of Step Size, depending on Start Position
        step_par.setLimits([0.2,abs(start_par.value())])
        start_par.sigValueChanged.connect(self.set_step_limits)

        # Always update number of steps, given by start pos and step size
        start_par.sigValueChanged.connect(self.show_steps)
        step_par.sigValueChanged.connect(self.show_steps)

    def get_sensor_size(self):
        return self._sensor_width, self._sensor_height

    def set_crop_limits(self, param, changes):
        maxW, maxH = self.get_sensor_size()
        for param, change, data in changes:
            path = self.par.childPath(param)
            par = self.par.param(path[0]).child(path[1])
            if path[2]=='Width':
                mx = maxW
                par.child('Xpos').setLimits([0,mx-par.child(path[2]).value()])
            if path[2]=='Height':
                mx = maxH
                par.child('Ypos').setLimits([0,mx-par.child(path[2]).value()])
            if path[2]=='Xpos':
                mx = maxW
                par.child('Width').setLimits([1,mx-par.child(path[2]).value()])
            if path[2]=='Ypos':
                mx = maxH
                par.child('Height').setLimits([1,mx-par.child(path[2]).value()])

    def get_crop_par(self):
        """ Get the crop parameters from parameter tree"""
        roi_par = self.par.param('ALLIED VISION CCD').child('Crop Image')
        xpos = roi_par.child('Xpos').value()
        ypos = roi_par.child('Ypos').value()
        width = roi_par.child('Width').value()
        height = roi_par.child('Height').value()
        return xpos, ypos, width, height

    def update_crop_param(self, roi):
        """Used as action when changing roi in roi window"""
        pos = roi.pos()
        size = roi.size()
        # Update the CROP parameters regarding region of interest
        roi_par = self.par.param('ALLIED VISION CCD').child('Crop Image')
        # Create even numbers. Odd numbers crash with some cameras
        # and make sure that offset and size stays in allowed range
        max_size = self.get_sensor_size()
        for i in range(2):
            if pos[i] < 0:
                pos[i] = 0
            if size[i] > max_size[i]:
                size[i] = max_size[i]
                pos[i] = 0
            if size[i]+pos[i] > max_size[i]:
                size[i] = max_size[i] - pos[i]
            pos[i] = round(pos[i]/2.)*2
            size[i] = round(size[i]/2.)*2
        roi_par.child('Xpos').setValue(int(pos[0]))
        roi_par.child('Ypos').setValue(int(pos[1]))
        roi_par.child('Width').setValue(int(size[0]))
        roi_par.child('Height').setValue(int(size[1]))

    def set_step_limits(self, param, val):
        step_par = self.par.param('Newport Stage').child('Step Size')
        step_par.setLimits([0.2,abs(val)])

    def set_start_pos_limits(self, param, val):
        start_pos = self.par.param('Newport Stage').child('Start Position')
        start_pos.setLimits([-val,-0.2])

    def show_pos(self, val):
        pos = self.par.param('Newport Stage').child('Position')
        pos.setValue(val)

    def show_steps(self, dummy):
        start_pos = self.par.param('Newport Stage').child('Start Position')
        step_size = self.par.param('Newport Stage').child('Step Size')
        val = int(round(2*abs(start_pos.value())/step_size.value()))

        num = self.par.param('Newport Stage').child('Number of steps')
        num.setValue(val)

    def print_par_changes(self, val: bool=True):
        if val:
            # Do print changes in parametertree
            self.par.sigTreeStateChanged.connect(self._change)
        else:
            self.par.sigTreeStateChanged.disconnect(self._change)

    def _change(self, param, changes):
        ## If anything changes in the parametertree, print a message
        for param, change, data in changes:
            path = self.par.childPath(param)
            if path is not None:
                child_name = '.'.join(path)
            else:
                child_name = param.name()
            print("tree changes:")
            print('  parameter: %s'% child_name)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')


if __name__ == "__main__":

    frog = FROG()
    frog.initialize()
    frog.close()
