"""
Model for the FROG setup

An example of how to run the code is found at the end of this file.

File name: frog.py
Author: Julian Krauth
Date created: 2019/12/02
Python Version: 3.7                     
"""

import os
import sys
import time
import numpy as np

cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_dir)
python_dir = os.path.join(cur_dir, "..", "..", "Python")
sys.path.append(python_dir)

from Controller import NEWPORT
from Model import acquisition

class FROG:

    def __init__(self, test=True):
        # Load the FROG devices (optional: virtual devices for testing)
        if test:
            self.stage = NEWPORT.SMC100DUMMY(port='/dev/ttyUSB0', dev_number=1)
        else:
            self.stage = NEWPORT.SMC100(port='/dev/ttyUSB0', dev_number=1)
        self.spect = acquisition.Spectrometer(test)

        self.measured_trace = None
        self.stop_measure = False
        

    def initialize(self, mode=0):
        self.stage.initialize()
        self.spect.initialize(mode)


    def measure(self, sig_progress, sig_measure, start_pos, max_meas, step_size):
        # Delete previously measured trace from memory.
        self.measured_trace = None
        # Move stage to Start Position and wait for end of movement
        self.stage.goto(start_pos)
        self.stage.wait_move_finish(1.)
        for i in range(max_meas):
            print("Loop...")
            # Move stage
            self.stage.goto(start_pos+i*step_size)
            self.stage.wait_move_finish(0.2)
            # Record spectrum
            y = self.spect.get_spectrum()
            # Create 2d frog-array to fill with data
            if i==0: 
                frog_array = np.zeros((len(y),max_meas))
            # Stitch data together
            frog_array[:,i] = y
            # Send data to plot
            sig_measure.emit(3, y)
            sig_measure.emit(2, frog_array)
            time.sleep(1)
            sig_progress.emit(i+1)
            if self.stop_measure: break
        if self.stop_measure:
            print("Measurement aborted!")
        else:
            # Save Frog trace as Instance attribute, it is then available for saving.
            self.measured_trace = frog_array
            print("Measurement finished!")

    def save_trace(self):
        pass
        
    def close(self):
        self.stage.close()
        self.spect.close()



if __name__ == "__main__":

    frog = FROG()
    frog.initialize()
    frog.close()