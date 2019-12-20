"""
This module loads the ROI window of the GUI

File name: roi_window.py
Author: Julian Krauth
Date created: 2019/12/20
Python Version: 3.7
"""

import pyqtgraph as pg

class ROIGraphics:
    """
    Class which creates the window for choosing the region of interest for the CCD camera.
    """
    def __init__(self):
        pass
        # Image will be filled with an image from the CCD
        #self.image = []

        
    def createWin(self):
        """
        Creates the window
        """

        # Create the window
        self.win = pg.GraphicsWindow()
        self.win.setWindowTitle('ROI - CCD')
        self.win.move(300,0)

        # Create two Viewboxes
        self.v1a = self.win.addViewBox(row=0, col=0, lockAspect=True)
        self.v1b = self.win.addViewBox(row=1, col=0, lockAspect=True)
        self.img1a = pg.ImageItem()
        self.v1a.addItem(self.img1a)
        self.img1b = pg.ImageItem()
        self.v1b.addItem(self.img1b)
        self.v1a.disableAutoRange('xy')
        self.v1b.disableAutoRange('xy')
        self.v1a.autoRange()
        self.v1b.autoRange()

        # Create ROI rectangle with arbitrary size, will later be updated
        self.roi = pg.RectROI([400, 470], [20, 10], pen=(0,9),
                              scaleSnap=True,translateSnap=True)
        
        self.roi.sigRegionChanged.connect(self.update)
        self.v1a.addItem(self.roi)

    def set_image(self, img):
        """Set the image in the viewport"""
        self.image = img
        self.img1a.setImage(self.image)
        self.v1a.autoRange()
        
    def update_ROI_frame(self,x,y,dx,dy):
        """Change position and size of ROI, don't send changefinished signal."""
        self.roi.setPos([x, y],finish=False)
        self.roi.setSize([dx, dy],finish=False)


    def update(self,roi):
        """Update the second viewport which shows the selection / ROI"""
        self.img1b.setImage(self.roi.getArrayRegion(self.image, self.img1a),
                            levels=(0, self.image.max()))
        self.v1b.autoRange()
  
    """ End ROIGraphics class """
