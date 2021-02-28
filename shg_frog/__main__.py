"""

*** The shg_frog software ***

Usage:
    shg_frog [-t]

Options:
    -t          run a dummy version, no devices, retrieval possible

"""
import sys
from docopt import docopt
from PyQt5.QtWidgets import QApplication

from .Model.frog import FROG
from .View.main_window import MainWindow

# Implement application execution options:
# Running a test mode with virtual devices

def main():
    args = docopt(__doc__)
    test_mode = bool(args['-t'])

    experiment = FROG(test=test_mode)

    app = QApplication(sys.argv)
    m = MainWindow(frog=experiment, test=test_mode)
    m.show()
    app.exit(app.exec_())

if __name__== "__main__":
    main()
