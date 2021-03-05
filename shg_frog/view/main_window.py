"""
This module loads the MainWindow of the GUI from a QTdesigner file
and connects all widgets to the methods of the devices.

File name: main_window.py
Author: Julian Krauth
Date created: 2019/12/02
Python Version: 3.7
"""
import pathlib
import numpy as np

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
            0: 'Camera',
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
        self.update_timer.setInterval(500) #ms
        self.update_timer.timeout.connect(self.update_values)

        # Connect button
        self.btn_connect.toggled.connect(self.connect_action)

        # Measure button
        self.btn_measure.toggled.connect(self.measure_action)

        # Save button
        self.btn_save.clicked.connect(self.save_action)

        # Load button
        self.btn_load.clicked.connect(self.load_action)

        # Create Parametertree from FrogParams class
        self.par_class = self.frog.parameters
        self.par = self.par_class.par
        self.print_changes(True)
        # Create ParameterTree widget filled with above parameters
        self.parameter_tree = ParameterTree()
        self.parameter_tree.setParameters(self.par, showTop=False)
        self.gridLayout.addWidget(self.parameter_tree, 1, 0, 1, 2)
        # Implement Actions for ParameterTree
        self.tree_stage_actions()
        self.tree_camera_actions()
        #  self.tree_ando_actions()

        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')

        # Create the plot window
        self.graphics_widget = FrogGraphics()
        self.gridLayout_2.addWidget(self.graphics_widget,1,0,1,3)

        # Create instance for region of interest (ROI) window
        # This window will be opened and closed by the class' methods
        self.window_roi = ROIGraphics()
        self.btn_roi.clicked.connect(self.roi_action)

        # Attribute for measurement thread
        self.measure_thread = None

        # Attribute for phase retrieval window and thread
        self.window_retrieval = None
        self.phase_thread = None

        # Phase retrieve button
        self.btn_phase.clicked.connect(self.phase_action)

    def print_changes(self, val: bool):
        """ Control whether parameter changes are printed. """
        self.par_class.print_par_changes(val)


    @QtCore.pyqtSlot(bool)
    def connect_action(self, checked):
        """Connect to devices selected in the dropdown menu.
        Adapt button color and text accordingly."""
        # Get dictionaries
        dev = self.DEFAULTS['dev']
        btn = self.DEFAULTS['btn_connect']
        col = self.DEFAULTS['btn_color']
        # Get dropdown position
        # index = self.dropdown.currentIndex() # not used at the moment.
        index = 0
        # Do GUI actions
        # self.dropdown.setEnabled(not checked) # only use once ANDO implemented
        if index==0:
            self.btn_roi.setEnabled(checked)
        self.btn_connect.setText(btn[checked])
        self.btn_connect.setStyleSheet(f"background-color:{col[checked]}")
        self.btn_measure.setEnabled(checked)
        # Open device and respective branch of parameter tree
        if checked:
            self.frog.initialize()
            self.par.param(dev[index]).show()
            self.par.param('Stage').show()
            self.update_timer.start()
        else:
            self.update_timer.stop()
            self.frog.close()
            self.par.param(dev[index]).hide()
            self.par.param('Stage').hide()
        # needed for updating par tree in GUI
        self.parameter_tree.setParameters(self.par, showTop=False)


    def tree_stage_actions(self):
        stage_par = self.par.param('Stage')
        # Stage Position
        go_par = stage_par.child('GoTo')
        go_par.sigValueChanged.connect(lambda _, val: self.frog.stage.move_abs(val))

    def tree_camera_actions(self):
        """ Connect camera functionality. """
        camera_par = self.par.param('Camera')
        expos_par = camera_par.child('Exposure')
        expos_par.sigValueChanged.connect(lambda _, val: self.frog.camera.set_exposure(val))
        gain_par = camera_par.child('Gain')
        gain_par.sigValueChanged.connect(lambda _, val: self.frog.camera.set_gain(val))
        crop_par = camera_par.child('Crop Image')
        crop_par.sigTreeStateChanged.connect(self.crop_action)
        tsource_par = camera_par.child('Trigger').child('Source')
        tsource_par.sigValueChanged.connect(lambda _, val: self.frog.camera.set_trig_source(val))

    # def tree_ando_actions(self):
    #     """ Connect ando functionality. """
    #     ando_par = self.par.param('ANDO Spectrometer')
    #     ctr_par = ando_par.child('Center')
    #     ctr_par.sigValueChanged.connect(lambda param, val: self.frog.ando.ctr(val))
    #     span_par = ando_par.child('Span')
    #     span_par.sigValueChanged.connect(lambda param, val: self.frog.ando.span(val))
    #     cw_par = ando_par.child('CW mode')
    #     cw_par.sigValueChanged.connect(lambda param, val: self.frog.ando.cw_mode(val))
    #     holdtime_par = ando_par.child('Rep. time')
    #     holdtime_par.sigValueChanged.connect(lambda param, val:self.frog.ando.peak_hold_mode(val))

    def crop_action(self, param, changes):
        """Define what happens when changing the crop/roi parameters in the parameter tree"""
        dictio = {'Width':'width','Height':'height',
                'Xpos':'offsetx','Ypos':'offsety'}
        for param, change, data in changes:
            if change=='value':
                self.frog.camera.img_format(**{dictio[param.name()]:data})
                #print dict[param.name()], data

    def roi_action(self):
        """Defines the actions when calling the ROI button"""
        # Create ROI window with a full image taken by the camera
        self.window_roi.show()
        self.window_roi.set_image(self.frog.camera.take_full_img())
        # Set the ROI frame according to the crop parameters in parameter tree
        self.window_roi.update_ROI_frame(*self.par_class.get_crop_par())
        # If ROI changes, update parameters, update_crop_param() makes sure that crop parameters
        # don't extend over edges of image. This means that the crop parameters which are set
        # can differ from the roi frame in the roi window. In a second step the roi frame is then
        # updated to reflect the actual crop parameters.
        self.window_roi.roi.sigRegionChangeFinished.connect(self.par_class.update_crop_param)
        self.par.sigTreeStateChanged.connect(\
            lambda param,changes: self.window_roi.update_ROI_frame(*self.par_class.get_crop_par()))


    @QtCore.pyqtSlot(bool)
    def measure_action(self, checked):
        """Executed when measure/stop button is pressed"""
        btn = self.DEFAULTS['btn_measure']
        self.btn_measure.setText(btn[checked])
        if checked:
            self.progress.setValue(0)
            # Do actual measurement loop (in separate thread)
            self.start_measure()
        if not checked:
            self.frog.stop_measure = True

    def start_measure(self):
        """Retrieves measurement settings and wraps the measure function
        into a thread. Then the signals are implemented."""
        # Create thread
        self.measure_thread = general_worker.MeasureThread(self.frog.measure)
        # Actions when measurement finishes
        self.measure_thread.finished.connect(self.measure_thread.deleteLater)
        self.measure_thread.finished.connect(self.del_mthread)
        self.measure_thread.finished.connect(self.uncheck_btn_measure)
        # Connect progress button with progress signal
        self.measure_thread.sig_progress.connect(self.modify_progress)
        # Connect plot update with measure signal
        self.measure_thread.sig_measure.connect(self.graphics_widget.update_graphics)
        # Run measurement
        self.measure_thread.start()

    def del_mthread(self):
        self.measure_thread = None

    def uncheck_btn_measure(self):
        self.btn_measure.setChecked(False)

    def save_action(self):
        """ Open dialog to request a comment, then save the data. """
        # Ask user for a measurement comment
        dialog = CommentDialog(self)
        if dialog.exec():
            comment = dialog.get_comment()
            self.frog.save_measurement_data(comment)
        else:
            print('Data not saved.')


    def load_action(self):
        """ Open dialog to choose a measurement data to load. """
        data_dir = pathlib.Path(__file__).parents[2] / 'Data'
        load_dir = QtWidgets.QFileDialog.getExistingDirectory(self, \
            'Choose measurement folder', str(data_dir))
        try:
            plot_data = self.frog.load_measurement_data(pathlib.Path(load_dir))
            self.graphics_widget.update_graphics(2, plot_data)
        except FileNotFoundError:
            print("Error: This directory does not contain the files " + \
                "with the correct file names.")

    def update_values(self):
        """Used for values which are continuously updated using QTimer"""
        pos_par = self.par.param('Stage').child('Position')
        pos_par.setValue(self.frog.stage.position)

    def modify_progress(self, iter_val):
        """For changing the progress bar, using an iteration value"""
        max_val = self.par.param('Stage').child('Number of steps').value()
        val = int(100*(float(iter_val)/float(max_val)))
        self.progress.setValue(val)

    def phase_action(self):
        if not self.frog.data_available:
            print('Error: No data for phase retrieval found.')
            return
        # Open retrieval window, if necessary close previous one to avoid warning.
        if self.window_retrieval is not None:
            # This avoids a warning when you start a retrieval while the
            # window is still open.
            self.window_retrieval.close()
        self.window_retrieval = RetrievalGraphics()
        self.window_retrieval.show()
        # Create thread
        self.phase_thread = general_worker.RetrievalThread(self.frog.retrieve_phase)
        # Actions when retrieval finishes
        self.phase_thread.finished.connect(self.phase_thread.deleteLater)
        self.phase_thread.finished.connect(self.del_pthread)
        # Connect signals
        self.phase_thread.sig_retdata.connect(self.window_retrieval.update_graphics)
        self.phase_thread.sig_retlabels.connect(self.window_retrieval.update_labels)
        self.phase_thread.sig_rettitles.connect(self.window_retrieval.update_title)
        self.phase_thread.sig_retaxis.connect(self.window_retrieval.set_axis)
        # Run phase retrieval
        self.phase_thread.start()

    def del_pthread(self):
        """Delete phase retrieval thread"""
        self.phase_thread = None
        print('Retrieval window closed!')


class FrogGraphics(pg.GraphicsLayoutWidget):
    """
    Class which implements the content for the graphics widget.
    It shows the recorded data and updates during measurement.
    """
    def __init__(self):
        super().__init__()
        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        pl = self.addPlot(title='Single Slice')
        pl.setLabel('bottom', "Frequency", units='AU')
        pl.setLabel('left', "Intensity", units='AU')
        self.plot1 = pl.plot()
        vb_full = self.addPlot(title='FROG Trace')
        vb_full.setLabel('bottom',"Time Delay", units='AU')
        vb_full.setLabel('left',"Frequency", units='AU')
        self.plot2 = pg.ImageItem()
        vb_full.addItem(self.plot2)

    def update_graphics(self, plot_num: int, data):
        """ Update single Slice and FROG trace plots in main window
        Arguments:
        plot_num -- 1: Slice plot
        plot_num -- 2: FROG plot
        """
        print(plot_num)
        #if plot_num==1:
        #    self.plot1.setData(np.sum(data,0))
        if plot_num==3:
            self.plot1.setData(data)
        if plot_num==2:
            data = np.flipud(data)
            self.plot2.setImage(data)



class CommentDialog(QtWidgets.QDialog):
    """ For adding a comment when saving the measurement. """
    def __init__(self, parent):
        """ Setting up the window. """
        super().__init__(parent=parent)
        self.setWindowTitle('Add a comment (optional)')
        self.setFixedSize(300, 80)
        dialog_layout = QtWidgets.QVBoxLayout()

        form_layout = QtWidgets.QFormLayout()
        self.comment = QtWidgets.QLineEdit()
        form_layout.addRow('Name:', self.comment)
        dialog_layout.addLayout(form_layout)

        standard_buttons = QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok
        self.button_box = QtWidgets.QDialogButtonBox(standard_buttons)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        #self.buttons.setStandardButtons(self.buttons)
        dialog_layout.addWidget(self.button_box)
        self.setLayout(dialog_layout)


    def get_comment(self) -> str:
        """ Return the comment """
        return self.comment.text()
