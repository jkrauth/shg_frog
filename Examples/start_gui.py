"""
This script creates an instance of the frog and hands it over
to the GUI.

File name: start_gui.py
Author: Julian Krauth
Date created: 2019/12/02
Python Version: 3.7
"""


import sys
import os
from PyQt5.QtWidgets import QApplication

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
PYTHON_DIR = os.path.join(CUR_DIR, "..", "Python")
sys.path.append(PYTHON_DIR)

from Model.frog import FROG
from View.main_window import MainWindow

test_option = 'test' in sys.argv
experiment = FROG(test=test_option)

app = QApplication(sys.argv)
m = MainWindow(frog=experiment)
m.show()
app.exit(app.exec_())
