"""

*** The shg_frog software ***

Usage:
    shg_frog [-tv]

Options:
    -t          run a dummy version, no devices, retrieval possible
    -v          verbose, changes in parametertree are printed.
"""
import sys
from docopt import docopt
from PyQt5.QtWidgets import QApplication

from .Model.frog import FROG
from .View.main_window import MainWindow

# Implement application execution options:
# Running a test mode with virtual devices

def main():
    """ Loop for the main window of the shg_frog. """
    args = docopt(__doc__)
    test_mode = bool(args['-t'])
    verbose = bool(args['-v'])

    experiment = FROG(test=test_mode)

    app = QApplication(sys.argv)
    win = MainWindow(frog=experiment, test=test_mode)
    win.print_changes(verbose)
    win.show()
    app.exit(app.exec())

if __name__== "__main__":
    main()
