from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal

import numpy as np

class WorkThread(QtCore.QThread):
    
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __del__(self):
        self.wait()

    def run(self):
        self.function(*self.args, **self.kwargs)
        return

class MeasureThread(WorkThread):

    # Signal for progress bar
    sig_progress = pyqtSignal(int)
    # Signal for plotting data in GUI
    sig_measure = pyqtSignal(int,np.ndarray)
    
    def __init__(self, function, *args, **kwargs):
        super().__init__(function, *args, **kwargs)
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.function(self.sig_progress, self.sig_measure, \
                      *self.args, **self.kwargs)
        return
