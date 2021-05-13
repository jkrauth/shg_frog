""" Module that defines the QThreads in which measurement
and phase retrieval run. """
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal

import numpy as np

class WorkThread(QtCore.QThread):
    """ Base class for the threads defined further down. """
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __del__(self):
        self.wait()

    def run(self):
        self.function(*self.args, **self.kwargs)

class MeasureThread(WorkThread):
    """ Thread used for the measurement process. """
    # Signal for progress bar
    sig_progress = pyqtSignal(int)
    # Signal for plotting data in GUI
    sig_measure = pyqtSignal(int, np.ndarray)

    def run(self):
        self.function(self.sig_progress, self.sig_measure, \
                      *self.args, **self.kwargs)

class RetrievalThread(WorkThread):
    """ Thread used for the phase retrieval process. """
    # Signal for update graphics
    sig_retdata = pyqtSignal(int,np.ndarray)
    # Signal for update labels
    sig_retlabels = pyqtSignal(list)
    # Signal for update title
    sig_rettitles = pyqtSignal(int,float)
    # Signal for setting axis
    sig_retaxis   = pyqtSignal(np.ndarray,np.ndarray)

    def run(self):
        self.function(self.sig_retdata, self.sig_retlabels, \
                      self.sig_rettitles, self.sig_retaxis, \
                      *self.args, **self.kwargs)
