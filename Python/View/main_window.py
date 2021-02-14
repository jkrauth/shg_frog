"""
This module loads the MainWindow of the GUI from a QTdesigner file
and connects all widgets to the methods of the devices.

File name: main_window.py
Author: Julian Krauth
Date created: 2019/12/02
Python Version: 3.7
"""
import pathlib
import yaml
import numpy as np
import imageio

from PyQt5 import QtWidgets, uic, QtCore, QtGui
from pyqtgraph.parametertree import ParameterTree
import pyqtgraph as pg

from . import general_worker
from .roi_window import ROIGraphics
from .retrieval_window import RetrievalGraphics

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


    def __init__(self, frog=None, parent=None, test: bool=False):
        super().__init__(parent)

        # The object which is connected to the window
        self.frog = frog

        # Loading the GUI created with QTdesigner
        gui_path = pathlib.Path(__file__).parent / 'GUI'
        uic.loadUi(gui_path / 'main_window.ui', self)

        # Change window title if running in test mode
        if test:
            self.setWindowTitle('SHG Frog (TEST)')

        # Set window icon
        self.setWindowIcon(QtGui.QIcon(str(gui_path / 'icon.png')))

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
        self.par_class = self.frog.parameters
        # Print changes of parameters throughout operation
        self.par_class.print_par_changes()
        self.par = self.par_class.par
        # Create ParameterTree widget filled with above parameters
        self.parameter_tree = ParameterTree()
        self.parameter_tree.setParameters(self.par, showTop=False)
        self.gridLayout.addWidget(self.parameter_tree,1,0,1,2)
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
        """Connect to devices selected in the dropdown menu.
        Adapt button color and text accordingly."""
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
        self.parameter_tree.setParameters(self.par, showTop=False)


    def tree_stage_actions(self):
        stage_par = self.par.param('Newport Stage')
        # Stage Position
        go_par = stage_par.child('GoTo')
        go_par.sigValueChanged.connect(lambda param, val: self.frog.stage.move_abs(val))

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
        tsource_par.sigValueChanged.connect(lambda param,val:self.frog.spect.trig_source(val))
        # ANDO Connections
        spect_par = self.par.param('ANDO Spectrometer')
        ctr_par = spect_par.child('Center')
        ctr_par.sigValueChanged.connect(lambda param,val:self.frog.spect.ctr(val))
        span_par = spect_par.child('Span')
        span_par.sigValueChanged.connect(lambda param,val:self.frog.spect.span(val))
        cw_par = spect_par.child('CW mode')
        cw_par.sigValueChanged.connect(lambda param,val:self.frog.spect.cw_mode(val))
        holdtime_par = spect_par.child('Rep. time')
        holdtime_par.sigValueChanged.connect(lambda param,val:self.frog.spect.peak_hold_mode(val))

    def crop_action(self, param, changes):
        """Define what happens when changing the crop/roi parameters in the parameter tree"""
        dictio = {'Width':'width','Height':'height',
                'Xpos':'offsetx','Ypos':'offsety'}
        for param, change, data in changes:
            if change=='value':
                self.frog.spect.img_format(**{dictio[param.name()]:data})
                #print dict[param.name()], data

    def roi_action(self):
        """Defines the actions when calling the ROI button"""
        # Create ROI window with a full image taken by the camera
        self.win_roi.createWin()
        self.win_roi.set_image(self.frog.spect.take_full_img())
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

    @staticmethod
    def get_unique_path(directory: pathlib.Path, name_pattern: str):
        counter = 0
        while True:
            counter += 1
            path = directory / name_pattern.format(counter)
            if not path.exists():
                return counter, path

    def save_action(self):
        if self.frog.measured_trace is not None:
            outfolder = self.frog.get_data_path()
            filename = self.frog.get_file_name()
            imagetype = self.frog.get_image_suffix()
            metatype = self.frog.get_meta_suffix()
            image_pattern = filename + "_{:03d}" + imagetype
            meta_pattern = filename + "_{:03d}" + metatype
            file_num, unique_path = self.get_unique_path(outfolder, image_pattern)
            if not unique_path.parent.exists():
                unique_path.parent.mkdir(parents=True, exist_ok=True)

            pix_format = self.par.param('ALLIED VISION CCD').child('PixelFormat').value()
            if pix_format == 'Mono8':
                bit_type = np.uint8
            elif pix_format == 'Mono12':
                bit_type = np.uint16
            # Save matrix as image with numbered filename
            imageio.imsave(unique_path, \
                self.frog.measured_trace.astype(bit_type))
            # Save settings
            # Get settings from frog instance
            settings = self.frog.used_settings
            # Add additional information
            settings['measurement number'] = file_num
            settings['center position'] = self.par.param('Newport Stage').child('Offset').value()
            # maybe add possibility to add a comment: settings['comment'] =
            # Create yaml settings file to the measurement, with numbered name

            with open(unique_path.parent / meta_pattern.format(file_num), 'w') as f:
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
