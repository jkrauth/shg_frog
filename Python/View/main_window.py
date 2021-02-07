"""
This module loads the MainWindow of the GUI from a QTdesigner file
and connects all widgets to the methods of the devices.

An example of how to run the code is found at the end of this file.

File name: main_window.py
Author: Julian Krauth
Date created: 2019/12/02
Python Version: 3.7
"""
import sys
import os
import yaml
import numpy as np
import imageio

from PyQt5 import QtWidgets, uic, QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph as pg

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
config_dir = os.path.join(CUR_DIR, "..", "..", "Config")
sys.path.append(CUR_DIR)

import general_worker
from roi_window import ROIGraphics
from retrieval_window import RetrievalGraphics

class MainWindow(QtWidgets.QMainWindow):
    """This is the main window of the GUI for the FROG interface.
    The window is designed with Qt Designer and loaded into this class.
    """

    DEFAULTS = {
        'dev': {
            0: 'ALLIED VISION CCD',
            1: 'ANDO Spectrometer',
        },
        'btn_connect': {
            True: 'Disconnect',
            False: 'Connect',
        },
        'btn_color': {
            True: 'rgb(239, 41, 41)',
            False: 'rgb(138, 226, 52)',
        },
        'btn_measure': {
            True: 'Stop',
            False: 'Measure',
        },
    }

    # Add defaults from config file
    with open(config_dir + '/config.yml','r') as f:
        CONFIG = yaml.load(f,Loader=yaml.FullLoader)
    DEFAULTS['data folder'] = CONFIG['data folder']
    DEFAULTS['file name'] = CONFIG['file name']
    DEFAULTS['file type'] = CONFIG['file type']

    def __init__(self, frog=None, parent=None, test=False):
        super().__init__(parent)

        # The object which is connected to the window
        self.frog = frog

        # Loading the GUI created with QTdesigner
        gui_path = os.path.dirname(__file__)
        uic.loadUi(os.path.join(gui_path, 'GUI/main_window.ui'), self)

        # Change window title if running in test mode
        if test:
            self.setWindowTitle('SHG Frog (TEST)')

        # Set window icon
        self.setWindowIcon(QtGui.QIcon(os.path.join(gui_path, 'GUI/icon.png')))

        # Timer used to update certain values with a fixed interval (Timer starts after connecting)
        self.update_timer = QtCore.QTimer()
        self.update_timer.setInterval(500) # 1000ms = 1s
        self.update_timer.timeout.connect(self.update_values)

        # Connect button
        self.btn_connect.toggled.connect(self.connect_action)

        # Measure button
        self.btn_measure.toggled.connect(self.measure_action)

        # Save button
        self.btn_save.clicked.connect(self.save_action)

        # Create Parametertree from FrogParams class
        self.par_class = FrogParams()
        # Print changes of parameters throughout operation
        self.par_class.printParChanges()
        self.par = self.par_class.par
        # Create ParameterTree widget filled with above parameters
        self.parTree = ParameterTree()
        self.parTree.setParameters(self.par, showTop=False)
        self.gridLayout.addWidget(self.parTree,1,0,1,2)
        # Implement Actions for ParameterTree
        self.tree_stage_actions()
        self.tree_spect_actions()

        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')

        # Create the plot window
        self.plot_class = FrogGraphics()
        self.graphics_widget = self.plot_class.gw
        self.gridLayout_2.addWidget(self.graphics_widget,1,0,1,3)

        # Create instance for region of interest (ROI) window
        # This window will be opened and closed by the class' methods
        self.win_roi = ROIGraphics()
        self.btn_roi.clicked.connect(self.roi_action)

        # Create phase retrieval window
        self.win_ret = RetrievalGraphics()

        # Phase retrieve button
        self.btn_phase.clicked.connect(self.phase_action)


    @QtCore.pyqtSlot(bool)
    def connect_action(self, checked):
        # Get dictionaries
        dev = self.DEFAULTS['dev']
        btn = self.DEFAULTS['btn_connect']
        col = self.DEFAULTS['btn_color']
        # Get dropdown position
        index = self.dropdown.currentIndex()
        # Do GUI actions
        self.dropdown.setEnabled(not checked)
        if index==0:
            self.btn_roi.setEnabled(checked)
        self.btn_connect.setText(btn[checked])
        self.btn_connect.setStyleSheet(f"background-color:{col[checked]}")
        self.btn_measure.setEnabled(checked)
        # Open device and respective branch of parameter tree
        if checked:
            self.frog.initialize(index)
            self.par.param(dev[index]).show()
            self.par.param('Newport Stage').show()
            self.update_timer.start()
        else:
            self.update_timer.stop()
            self.frog.close()
            self.par.param(dev[index]).hide()
            self.par.param('Newport Stage').hide()
        # needed for updating par tree in GUI
        self.parTree.setParameters(self.par, showTop=False)


    def tree_stage_actions(self):
        stage_par = self.par.param('Newport Stage')
        # Stage Position
        go_par = stage_par.child('GoTo')
        go_par.sigValueChanged.connect(lambda param, val: self.frog.stage.goto(val))

    def tree_spect_actions(self):
        # Camera connections
        spect_par = self.par.param('ALLIED VISION CCD')
        expos_par = spect_par.child('Exposure')
        expos_par.sigValueChanged.connect(lambda param,val:self.frog.spect.exposure(val))
        gain_par = spect_par.child('Gain')
        gain_par.sigValueChanged.connect(lambda param,val:self.frog.spect.gain(val))
        crop_par = spect_par.child('Crop Image')
        crop_par.sigTreeStateChanged.connect(self.crop_action)
        tsource_par = spect_par.child('Trigger').child('Source')
        tsource_par.sigValueChanged.connect(lambda param,val:self.frog.spect.trigSource(val))
        # ANDO Connections
        spect_par = self.par.param('ANDO Spectrometer')
        ctr_par = spect_par.child('Center')
        ctr_par.sigValueChanged.connect(lambda param,val:self.frog.spect.ctr(val))
        span_par = spect_par.child('Span')
        span_par.sigValueChanged.connect(lambda param,val:self.frog.spect.span(val))
        cw_par = spect_par.child('CW mode')
        cw_par.sigValueChanged.connect(lambda param,val:self.frog.spect.cwMode(val))
        holdtime_par = spect_par.child('Rep. time')
        holdtime_par.sigValueChanged.connect(lambda param,val:self.frog.spect.peakHoldMode(val))

    def crop_action(self, param, changes):
        """Define what happens when changing the crop/roi parameters in the parameter tree"""
        dictio = {'Width':'width','Height':'height',
                'Xpos':'offsetx','Ypos':'offsety'}
        for param, change, data in changes:
            if change=='value':
                self.frog.spect.imgFormat(**{dictio[param.name()]:data})
                #print dict[param.name()], data

    def roi_action(self):
        """Defines the actions when calling the ROI button"""
        # Create ROI window with a full image taken by the camera
        self.win_roi.createWin()
        self.win_roi.set_image(self.frog.spect.takeFullImg())
        # Set the ROI frame according to the crop parameters in parameter tree
        self.win_roi.update_ROI_frame(*self.par_class.get_crop_par())
        # If ROI changes, update parameters, update_crop_param() makes sure that crop parameters
        # don't extend over edges of image. This means that the crop parameters which are set
        # can differ from the roi frame in the roi window. In a second step the roi frame is then
        # updated to reflect the actual crop parameters.
        self.win_roi.roi.sigRegionChangeFinished.connect(self.par_class.update_crop_param)
        self.par.sigTreeStateChanged.connect(\
            lambda param,changes: self.win_roi.update_ROI_frame(*self.par_class.get_crop_par()))


    @QtCore.pyqtSlot(bool)
    def measure_action(self, checked):
        btn = self.DEFAULTS['btn_measure']
        self.btn_measure.setText(btn[checked])
        if checked:
            self.progress.setValue(0)
            self.frog.stop_measure = False
            # Do actual measurement loop (in separate thread)
            self.start_measure()
        if not checked:
            self.frog.stop_measure = True

    def start_measure(self):
        """Retrieves measurement settings and wraps the measure function
        into a thread. Then the signals are implemented."""
        # Get settings
        start_pos = self.par.param('Newport Stage').child('Start Position').value()
        max_meas = self.par.param('Newport Stage').child('Number of steps').value()
        step_size = self.par.param('Newport Stage').child('Step Size').value()
        # Create thread
        self.measure_thread = general_worker.MeasureThread(self.frog.measure, \
            start_pos, max_meas, step_size)
        # Actions when measurement finishes
        self.measure_thread.finished.connect(self.measure_thread.deleteLater)
        self.measure_thread.finished.connect(self.del_mthread)
        self.measure_thread.finished.connect(self.automatic_toggle)
        # Connect progress button with progress signal
        self.measure_thread.sig_progress.connect(self.modifyProgress)
        # Connect plot update with measure signal
        self.measure_thread.sig_measure.connect(self.plot_class.updateGraphics)
        # Run measurement
        self.measure_thread.start()

    def del_mthread(self):
        """Delete measure_thread."""
        del self.measure_thread

    def automatic_toggle(self):
        """Toggles measurement button if measurement finished without manual stop."""
        if not self.frog.stop_measure:
            self.btn_measure.toggle()
        else:
            pass

    def save_action(self):
        if self.frog.measured_trace is not None:
            outfolder = self.DEFAULTS['data folder']
            filename = self.DEFAULTS['file name']
            filetype = self.DEFAULTS['file type']
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)
            filename_numbered = filename + "_{:03}"
            save_path = outfolder + filename_numbered
            file_num = 1
            while os.path.exists(save_path.format(file_num) + filetype):
                file_num += 1
            pix_format = self.par.param('ALLIED VISION CCD').child('PixelFormat').value()
            if pix_format == 'Mono8':
                bit_type = np.uint8
            elif pix_format == 'Mono12':
                bit_type = np.uint16
            # Save matrix as image with numbered filename
            imageio.imsave(save_path.format(file_num) + filetype, \
                        self.frog.measured_trace.astype(bit_type))
            # Save settings
            # Get settings from frog instance
            settings = self.frog.used_settings
            # Add additional information
            settings['measurement number'] = file_num
            settings['center position'] = self.par.param('Newport Stage').child('Offset').value()
            # maybe add possibility to add a comment: settings['comment'] =
            # Create yaml settings file to the measurement, with numbered name
            with open('%s.yml' % (save_path.format(file_num)), 'w') as f:
                f.write(yaml.dump(settings, default_flow_style=False))
            print('Measurement and settings saved!')
        else:
            print('Do measurement first!')


    def update_values(self):
        """Used for values which are continuously updated using QTimer"""
        pos_par = self.par.param('Newport Stage').child('Position')
        pos_par.setValue(self.frog.stage.position)

    def modifyProgress(self, iter_val):
        """For changing the progress bar, using an iteration value"""
        max_val = self.par.param('Newport Stage').child('Number of steps').value()
        val = int(100*(float(iter_val)/float(max_val)))
        self.progress.setValue(val)

    def phase_action(self):
        # Open retrieval window
        self.win_ret.createWin()
        # Call retrieval parameters
        pixels = self.par.param('Phase Retrieval').child('prepFROG Size').value()
        gtol = self.par.param('Phase Retrieval').child('G Tolerance').value()
        itermax = self.par.param('Phase Retrieval').child('Max. Iterations').value()
        # Create thread
        self.phase_thread = general_worker.RetrievalThread(self.frog.retrieve_phase, \
            pixels, gtol, itermax)
        # Actions when retrieval finishes
        self.phase_thread.finished.connect(self.phase_thread.deleteLater)
        self.phase_thread.finished.connect(self.del_pthread)
        # Connect signals
        self.phase_thread.sig_retdata.connect(self.win_ret.updateGraphics)
        self.phase_thread.sig_retlabels.connect(self.win_ret.updateLabels)
        self.phase_thread.sig_rettitles.connect(self.win_ret.updateTitle)
        self.phase_thread.sig_retaxis.connect(self.win_ret.setAxis)
        # Run phase retrieval
        self.phase_thread.start()

    def del_pthread(self):
        """Delete phase retrieval thread"""
        del self.phase_thread



class FrogParams:
    """
    Class which implements the parameters used by the parametertree widget.
    """
    # Load defaults:
    with open(config_dir + '/config.yml','r') as f:
        CONFIG = yaml.load(f,Loader=yaml.FullLoader)
    DEFAULTS = {'maxW': CONFIG['pxls width'],
                'maxH': CONFIG['pxls height'],
    }

    def __init__(self):
        # Define parameters for parametertree
        params = [
            {'name':'Phase Retrieval','type':'group','visible':True,
             'children': [
                 {'name':'prepFROG Size', 'type': 'int', 'value': 128},
                 {'name':'Seed', 'type': 'list', 'values': {"Random":0,"fromFile":1}, 'value':0},
                 {'name':'Max. Iterations', 'type': 'int', 'value': 200},
                 {'name':'G Tolerance', 'type': 'float', 'value': 0.001}
             ]},
            {'name':'ANDO Spectrometer','type':'group','visible':False,
             'children': [
                 {'name':'Center', 'type': 'float', 'value': 390.},
                 {'name':'Span', 'type': 'float', 'value': 20.},
                 {'name':'CW mode', 'type': 'bool', 'value': True},
                 {'name':'Rep. time', 'type': 'float', 'value': 36},
                 {'name':'Sampling', 'type': 'int', 'value': 201}
             ]},
            {'name':'ALLIED VISION CCD','type':'group','visible':False,
             'children': [
                 {'name':'Exposure', 'type': 'float', 'value': 0.036, 'dec': True, 'step': 1, 'siPrefix': True, 'suffix': 's'},
                 {'name':'Gain', 'type': 'float', 'value': 0, 'dec': False, 'step': 1},
                 {'name':'Crop Image', 'type': 'group', 'expanded':False,
                  'children': [
                     {'name':'Width','type':'int',
                      'value': 600,'limits':[1,self.DEFAULTS['maxW']],'step': 2},
                     {'name':'Height','type':'int',
                      'value': 10,'limits':[1,self.DEFAULTS['maxH']],'step': 2},
                     {'name':'Xpos','type':'int','value': 400,'step': 2},
                     {'name':'Ypos','type':'int','value': 470,'step': 2}
                 ]},
                 {'name':'Trigger', 'type': 'group', 'children': [
                     {'name':'Mode','type':'list','visible':False,
                      'values': {"On":1,"Off":0},'value':1},
                     {'name':'Source','type':'list','visible':True,
                      'values': {"Freerun":'Freerun',
                                 "External":'Line1'
                                 #'Line2':'Line2',
                                 #'FixedRate':'FixedRate',
                                 #'Software':'Software'
                      },'value':'External'}
                 ]},
                 {'name':'PixelFormat', 'type': 'list',
                  'values': {'Mono8':'Mono8','Mono12':'Mono12'
                             #'Mono12Packed':'Mono12Packed'
                  },
                  'value':'Mono8'},
             ]},
            {'name':'Newport Stage','type':'group','visible':False,
             'children': [
                 {'name':'Position','type':'float','value': 0., 'readonly': True},
                 {'name':'GoTo','type':'float','value': 0.},
                 {'name':'Offset','type':'float','value': 11370,
                  'limits':[0,25000]},
                 {'name':'Start Position','type':'float','value': -256},
                 {'name':'Step Size','type':'float','value': 4.},
                 {'name':'Number of steps','type':'int','readonly':True,
                  'value': 128}
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
        maxW = self.DEFAULTS['maxW']
        maxH = self.DEFAULTS['maxH']
        width_par.setLimits([1,maxW-xpos_par.value()])
        height_par.setLimits([1,maxH-ypos_par.value()])
        xpos_par.setLimits([0,maxW-width_par.value()])
        ypos_par.setLimits([0,maxH-height_par.value()])
        crop_par.sigTreeStateChanged.connect(self.setCropLimits)

        ### Some settings regarding the Stage parameters ###
        stage_par = self.par.param('Newport Stage')
        start_par = stage_par.child('Start Position')
        step_par = stage_par.child('Step Size')
        off_par = stage_par.child('Offset')

        # Set limits of Start Position, depending on offset
        start_par.setLimits([-off_par.value(),-0.2])
        off_par.sigValueChanged.connect(self.setStartPosLimits)

        # Set limits of Step Size, depending on Start Position
        step_par.setLimits([0.2,abs(start_par.value())])
        start_par.sigValueChanged.connect(self.setStepLimits)

        # Always update number of steps, given by start pos and step size
        start_par.sigValueChanged.connect(self.showSteps)
        step_par.sigValueChanged.connect(self.showSteps)

    def setCropLimits(self, param, changes):
        maxW = self.DEFAULTS['maxW']
        maxH = self.DEFAULTS['maxH']
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
        maxSize = [self.DEFAULTS['maxW'], self.DEFAULTS['maxH']]
        for i in range(2):
            if pos[i] < 0:
                pos[i] = 0
            if size[i] > maxSize[i]:
                size[i] = maxSize[i]
                pos[i] = 0
            if size[i]+pos[i] > maxSize[i]:
                size[i] = maxSize[i] - pos[i]
            pos[i] = round(pos[i]/2.)*2
            size[i] = round(size[i]/2.)*2
        roi_par.child('Xpos').setValue(int(pos[0]))
        roi_par.child('Ypos').setValue(int(pos[1]))
        roi_par.child('Width').setValue(int(size[0]))
        roi_par.child('Height').setValue(int(size[1]))

    def setStepLimits(self, param, val):
        step_par = self.par.param('Newport Stage').child('Step Size')
        step_par.setLimits([0.2,abs(val)])

    def setStartPosLimits(self, param, val):
        start_pos = self.par.param('Newport Stage').child('Start Position')
        start_pos.setLimits([-val,-0.2])

    def showPos(self, val):
        pos = self.par.param('Newport Stage').child('Position')
        pos.setValue(val)

    def showSteps(self, dummy):
        start_pos = self.par.param('Newport Stage').child('Start Position')
        step_size = self.par.param('Newport Stage').child('Step Size')
        val = int(round(2*abs(start_pos.value())/step_size.value()))

        num = self.par.param('Newport Stage').child('Number of steps')
        num.setValue(val)

    def printParChanges(self):
        # Do print changes in parametertree
        self.par.sigTreeStateChanged.connect(self._Change)

    def _Change(self, param, changes):
        ## If anything changes in the parametertree, print a message
        for param, change, data in changes:
            path = self.par.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()
            print("tree changes:")
            print('  parameter: %s'% childName)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')

    """ End FrogParams class """




class FrogGraphics:
    """
    Class which implements the content for the graphics widget.
    It shows the recorded data and updates during measurement.
    """
    def __init__(self):

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        self.gw = pg.GraphicsLayoutWidget()
        pl = self.gw.addPlot(title='Single Slice')
        pl.setLabel('bottom', "Frequency", units='AU')
        pl.setLabel('left', "Intensity", units='AU')
        self.plot1 = pl.plot()
        vb_full = self.gw.addPlot(title='FROG Trace')
        vb_full.setLabel('bottom',"Time Delay", units='AU')
        vb_full.setLabel('left',"Frequency", units='AU')
        self.plot2 = pg.ImageItem()
        vb_full.addItem(self.plot2)

    def updateGraphics(self,plot_num,data):
        if plot_num==1:
            self.plot1.setData(np.sum(data,0))
        if plot_num==3:
            self.plot1.setData(data)
        if plot_num==2:
            data = np.flipud(data)
            self.plot2.setImage(data)

    """ End FROG graphics class """

