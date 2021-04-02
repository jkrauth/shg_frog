"""
This module loads the retrieval window of the GUI

File name: retrieval_window.py
Author: Julian Krauth
Date created: 2020/01/17
Python Version: 3.7
"""
import sys
import pathlib
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    )
import pyqtgraph as pg
import numpy as np
import imageio
from matplotlib import cm


AMPLITUDE_COLOR = (255, 178, 102)
PHASE_COLOR = (102, 178, 255)
AMPLITUDE_LABEL = 'Amplitude'
PHASE_LABEL = 'Phase'

class RetrievalWindow(QWidget):
    """ This is the window for the phase retrieval. It contains
    the plots and some buttons. """
    def __init__(self, algo=None):
        """ Initializer """
        super().__init__()
        self.algo = algo
        self.setWindowTitle('SHG FROG - Phase Retrieval')
        self.setFixedSize(1000, 1000)
        # self.setStyleSheet("background:black")
        self.main_layout = QVBoxLayout()
        self._create_graphics()
        self._create_buttons()
        self._connect_buttons()
        self.setLayout(self.main_layout)

    def _create_graphics(self):
        self.graphics = RetrievalGraphics()
        self.main_layout.addWidget(self.graphics)

    def _create_buttons(self):
        self.btn_save_seed = QPushButton("Save as Seed")
        self.btn_save_data = QPushButton("Save Reconstructed Data")
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_save_seed)
        button_layout.addWidget(self.btn_save_data)
        self.main_layout.addLayout(button_layout)

    def _connect_buttons(self):
        if self.algo is not None:
            self.btn_save_seed.clicked.connect(self.algo.save_pulse_as_seed)
            self.btn_save_data.clicked.connect(self.save_data)

    def save_data(self):
        """ Save reconstructed data to path chosen by user. """
        data_path = QFileDialog.getExistingDirectory(
            self, "Select folder to save data", str(pathlib.Path.home())
        )
        if data_path:
            data_path = pathlib.Path(data_path)

            header, function_data = self.graphics.get_function_data()
            self._save_function_data(data_path / "function_data.txt", header, function_data)

            frog_orig, frog_reconstr = self.graphics.get_image_data()
            self._save_image_data(data_path / "original.png", frog_orig)
            self._save_image_data(data_path / "reconstructed.png", frog_reconstr)
            print("Data saved to " + str(data_path))

    @staticmethod
    def _save_function_data(file_path: pathlib.Path, header: str, function_data: list):
        np.savetxt(
            file_path,
            np.column_stack(function_data),
            header=header
        )

    @staticmethod
    def _save_image_data(file_path: pathlib.Path, image_data: np.ndarray):
        image = np.round(image_data*255).astype(np.uint8)
        imageio.imwrite(file_path, image)


class RetrievalGraphics(pg.GraphicsLayoutWidget):
    """
    Class which defines the graphics widget for the phase retrieval.
    """
    def __init__(self):
        """ Set up everything """
        super().__init__()

        self._create_plots()
        # Colormap for FROG images
        self._set_colormap('plasma')

        self.tpxls = None
        self.vpxls = None
        self.dvunit = None
        self.dtunit = None

    def _create_plots(self):
        # Add item to show original trace
        self.plot_original = self.addPlot(title='Orig. FROG trace')
        self.img_original = pg.ImageItem()
        self.plot_original.addItem(self.img_original)
        self.plot_original.setLabel('bottom', 'Delay')
        self.plot_original.setLabel('left', 'SH freq')

        # Add item to show retrieved trace
         # title is updated dynamically later
        self.plot_reconstructed = self.addPlot(title='Reconstructed')
        self.img_reconstructed = pg.ImageItem()
        self.plot_reconstructed.addItem(self.img_reconstructed)
        self.plot_reconstructed.setLabel('bottom', 'Delay')
        self.plot_reconstructed.setLabel('left', 'SH freq')
        # Link scale/shift between plots plot_original and plot_reconstructed
        self.plot_reconstructed.setXLink(self.plot_original)
        self.plot_reconstructed.setYLink(self.plot_original)

        y_label = '|E|^2 & ang(E)'
        # Add item to show pulse in time domain
        self.nextRow()
        self.plot_time = self.addPlot(colspan=2)
        self.plot_time.setLabel('left', y_label)
        self.plot_time.addLegend()
        self.plot_time.setLabel('bottom', 'Time')

        # Add item to show pulse in frequency domain
        self.nextRow()
        self.plot_freq = self.addPlot(colspan=2)
        self.plot_freq.setLabel('left', y_label)
        self.plot_freq.addLegend()
        self.plot_freq.setLabel('bottom', 'Frequency')
        #self.plot_freq.setXLink(self.plot_time)
        #self.plot_freq.setYLink(self.plot_time)


    def _set_colormap(self, color_map_name: str):
        """ Sets a colormap from matplotlib for the FROG images.
        Arguments:
        color_map_name -- str, name of one of the many color maps available
                          from matplotlib.
        """
        colormap = cm.get_cmap(color_map_name)
        colormap._init()
        # Convert matplotlib colormap from 0-1 to 0-255 for Qt
        lut = (colormap._lut * 255).view(np.ndarray)
        lut = lut[1:-3] # Truncate array, for some reason it is too long
        # Apply the colormap
        self.img_original.setLookupTable(lut)
        self.img_reconstructed.setLookupTable(lut)

    def set_axis(self, tpxls: np.ndarray, vpxls: np.ndarray):
        """
        Sets axis attributes which are needed for x and y scale
        """
        N = len(tpxls)
        # Set attributes which are needed to update plots later
        self.tpxls = tpxls
        self.vpxls = vpxls
        # Set correct scalings for axis of FROG traces
        self.img_original.scale(tpxls[1]-tpxls[0], vpxls[1]-vpxls[0])
        self.img_reconstructed.scale(tpxls[1]-tpxls[0], vpxls[1]-vpxls[0])
        self.img_original.translate(-N/2,-N/2) # center axes
        self.img_reconstructed.translate(-N/2,-N/2) # center axes
        # Create initial graphs for showing the retrieved pulse
        self.func_ampl_time = self.plot_time.plot(
            tpxls, np.zeros(N), pen=AMPLITUDE_COLOR, name=AMPLITUDE_LABEL)
        self.func_phase_time = self.plot_time.plot(
            tpxls, np.zeros(N), pen=PHASE_COLOR, name=PHASE_LABEL)
        self.func_ampl_freq = self.plot_freq.plot(
            vpxls, np.zeros(N), pen=AMPLITUDE_COLOR, name=AMPLITUDE_LABEL)
        self.func_phase_freq = self.plot_freq.plot(
            vpxls, np.zeros(N), pen=PHASE_COLOR, name=PHASE_LABEL)


    def update_graphics(self, which: int, data: np.ndarray):
        """
        Can only be used after window has been created and
        axes set by method set_axis()
        """
        if which==0: # Set original FROG trace
            self.img_original.setImage(data)
        if which==1: # Set reconstructed FROG trace
            self.img_reconstructed.setImage(data)
        if which==2: # Set pulse time amplitude
            self.func_ampl_time.setData(self.tpxls, data)
        if which==3: # Set pulse time phase
            self.func_phase_time.setData(self.tpxls, data)
        if which==4: # Set pulse freq amplitude
            self.func_ampl_freq.setData(self.vpxls, data)
        if which==5: # Set pulse freq phase
            self.func_phase_freq.setData(self.vpxls, data)


    def update_labels(self, units: list):
        """ Update the labels with the corresponding units. """
        self.dtunit = units[0]
        self.dvunit = units[1]
        self.plot_original.setLabel('bottom', 'Delay [%s]' % self.dtunit)
        self.plot_original.setLabel('left', 'SH freq [%s]' % self.dvunit)
        self.plot_reconstructed.setLabel('bottom', 'Delay [%s]' % self.dtunit)
        self.plot_reconstructed.setLabel('left', 'SH freq [%s]' % self.dvunit)
        self.plot_time.setLabel('bottom', 'Time [%s]' % self.dtunit)
        self.plot_freq.setLabel('bottom', 'Frequency [%s]' % self.dvunit)


    def update_title(self, iteration: int, tolerance: float):
        """ Update the title to set the current iteration and G values. """
        self.plot_reconstructed.setTitle(
            title=f'Reconstructed: iter={iteration:3} G={tolerance:.4f}'
            )

    def get_function_data(self):
        """ Returns the arrays that comprise the reconstructed pulse traces. """
        time, ampl_t = self.func_ampl_time.getData()
        _, phase_t = self.func_phase_time.getData()
        freq, ampl_f = self.func_ampl_freq.getData()
        _, phase_f = self.func_phase_freq.getData()
        header = ("time [%s], ampl_t [au], phase_t [mrad], "
            "frequency [%s], ampl_f [au], phase_f [mrad]") % (self.dtunit, self.dvunit)
        return header, [time, ampl_t, phase_t, freq, ampl_f, phase_f]

    def get_image_data(self):
        """ Returns the prepared original FROG image and the reconstructed trace. """
        original = self.img_original.image
        reconstructed = self.img_reconstructed.image
        return original, reconstructed


    # def screenshot(self, widget):
    #     exporter = pg.exporters.ImageExporter(widget)
    #     #exporter.parameters()['widht'] = 100
    #     exporter.export('screenshot.png')

if __name__ == "__main__":

    app = QApplication(sys.argv)
    win = RetrievalWindow()
    win.show()
    sys.exit(app.exec())
