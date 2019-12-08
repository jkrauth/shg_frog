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

from PyQt5 import QtWidgets, uic, QtCore, QtGui

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(CUR_DIR)

class MainWindow(QtWidgets.QMainWindow):
    """This is the main window of the GUI for the FROG interface.
    The window is designed with Qt Designer and loaded into this class.
    """
    def __init__(self, frog=None, parent=None):
        super().__init__(parent)

        # The object which is connected to the window
        self.frog = frog

        # Loading the GUI created with QTdesigner
        gui_path = os.path.dirname(__file__)
        uic.loadUi(os.path.join(gui_path, 'GUI/main_window.ui'), self)

        self.btn_connect.toggled.connect(self.connect_action)

    @QtCore.pyqtSlot(bool)
    def connect_action(self, checked):
        if checked:
            self.frog.initialize(self.dropdown.currentIndex())
        else:
            self.frog.close(self.dropdown.currentIndex())
            
if __name__ == "__main__":

    import sys
    

    app = QtGui.QApplication([])
    win = MainWindow()
