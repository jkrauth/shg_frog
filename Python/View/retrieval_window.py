"""
This module loads the retrieval window of the GUI

File name: retrieval_window.py
Author: Julian Krauth
Date created: 2020/01/17
Python Version: 3.7
"""
import numpy as np
import pyqtgraph as pg
from matplotlib import cm

class RetrievalGraphics:
    """
    Class which creates the graphics output window for the phase retrieval.
    """
    def __init__(self):
        pass

    def createWin(self):
        """
        Creates the window
        """
        # Create the window
        self.win = pg.GraphicsWindow()
        self.win.setWindowTitle('Phase Retrieval - SHG FROG')
        self.win.setGeometry(400,0,1000,1000)
        #self.win.move(300,0)


        # Add item to show original trace
        self.p1 = self.win.addPlot(title='Orig. FROG trace')
        self.img1 = pg.ImageItem()
        self.p1.addItem(self.img1)

        # Add item to show retrieved trace
        self.p2 = self.win.addPlot() # title is added later, will be dynamic
        self.img2 = pg.ImageItem()
        self.p2.addItem(self.img2)
        # Link scale/shift between plots p1 and p2
        self.p2.setXLink(self.p1)
        self.p2.setYLink(self.p1)

        # Add item to show pulse in time domain
        self.win.nextRow()
        self.p3 = self.win.addPlot(colspan=2)
        self.p3.setLabel('left','|E|^2 & ang(E)')

        # Add item to show pulse in frequency domain
        self.win.nextRow()
        self.p4 = self.win.addPlot(colspan=2)
        self.p4.setLabel('left','|E|^2 & ang(E)')
        #self.p4.setXLink(self.p3)
        #self.p4.setYLink(self.p3)

        # Create Colormaps from matplotlib colormaps
        if 1:
            colormap = cm.get_cmap("plasma")
            colormap._init()
            # Convert matplotlib colormap from 0-1 to 0-255 for Qt
            lut = (colormap._lut * 255).view(np.ndarray)
            lut = lut[1:-3] # Truncate array, for some reason it is too long
            # Apply the colormap
            self.img1.setLookupTable(lut)
            self.img2.setLookupTable(lut)


    #@pyqtSlot(np.ndarray,np.ndarray)
    def setAxis(self,tpxls,vpxls):
        """
        Sets axis attributes which are needed for x and y scale
        """
        N = len(tpxls)
        # Set attributes which are needed to update plots later
        self.tpxls = tpxls
        self.vpxls = vpxls
        # Set correct scalings for axis of FROG traces
        self.img1.scale(tpxls[1]-tpxls[0],vpxls[1]-vpxls[0])
        self.img2.scale(tpxls[1]-tpxls[0],vpxls[1]-vpxls[0])
        self.img1.translate(-N/2,-N/2) # center axes
        self.img2.translate(-N/2,-N/2) # center axes
        # Create initial graphs for showing the retrieved pulse
        self.p3p = self.p3.plot(tpxls,np.zeros(N),pen=(255,0,0))
        self.p3p2= self.p3.plot(tpxls,np.zeros(N),pen=(0,255,0))
        self.p4p = self.p4.plot(vpxls,np.zeros(N),pen=(255,0,0))
        self.p4p2= self.p4.plot(vpxls,np.zeros(N),pen=(0,255,0))


    #@pyqtSlot(int,np.ndarray)
    def updateGraphics(self,which,data):
        """
        Can only be used after window has been created and
        axes set by method setAxis()
        """
        if which==0: # Set original FROG trace
            self.img1.setImage(data)
        if which==1: # Set reconstructed FROG trace
            self.img2.setImage(data)
        if which==2: # Set pulse time data
            self.p3p.setData(self.tpxls,data)
        if which==3: # Set pulse time phase
            self.p3p2.setData(self.tpxls,data)
        if which==4: # Set pulse freq data
            self.p4p.setData(self.vpxls,data)
        if which==5: # Set pulse time phase
            self.p4p2.setData(self.vpxls,data)


    #@pyqtSlot(list)
    def updateLabels(self,units):
        dtunit = units[0]
        dvunit = units[1]
        self.p1.setLabel('bottom','Delay [%s]' % dtunit)
        self.p1.setLabel('left','SH freq [%s]' % dvunit)
        self.p2.setLabel('bottom','Delay [%s]' % dtunit)
        self.p2.setLabel('left','SH freq [%s]' % dvunit)
        self.p3.setLabel('bottom','Time [%s]' % dtunit)
        self.p4.setLabel('bottom','Frequency [%s]' % dvunit)

    #@pyqtSlot(int,float)
    def updateTitle(self,iteration,G):
        self.p2.setTitle(title='Reconstructed: iter=%d G=%.4f'
                         % (iteration,G))

    def screenshot(self,widget):
        exporter = pg.exporters.ImageExporter(widget)
        #exporter.parameters()['widht'] = 100
        exporter.export('screenshot.png')

    """ End RetrievalGraphics class """
