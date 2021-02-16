"""
This script creates an instance of the frog module and hands it over
to the GUI.

Usage:
$ python start_frog.py <option>

If option <option> = test is provided, the program is run with virtual devices.

File name: start_frog.py
Author: Julian Krauth
Date created: 2019/12/02
Python Version: 3.7
"""
import sys
from PyQt5.QtWidgets import QApplication

from Python.Model.frog import FROG
from Python.View.main_window import MainWindow

# Implement application execution options:
# Running a test mode with virtual devices
test_mode = 'test' in sys.argv

experiment = FROG(test=test_mode)

app = QApplication(sys.argv)
m = MainWindow(frog=experiment, test=test_mode)
m.show()
app.exit(app.exec_())
