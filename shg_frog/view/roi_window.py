"""
This module loads the ROI window of the GUI

File name: roi_window.py
Author: Julian Krauth
Date created: 2019/12/20
Python Version: 3.7
"""
import pyqtgraph as pg

class ROIGraphics(pg.GraphicsLayoutWidget):
    """
    Class which defines the window for choosing the region of interest for the CCD camera.
    """
    def __init__(self):
        """ Setting up everything """
        super().__init__()
        # Create the window
        self.setWindowTitle('ROI - CCD')
        self.move(300,0)

        # Create two Viewboxes
        self.v1b = self.addViewBox(row=1, col=0, lockAspect=True)
        self.v1a = self.addViewBox(row=0, col=0, lockAspect=True)
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

    def update_ROI_frame(self, x: int, y: int, dx: int, dy: int):
        """Change position and size of ROI, don't send changefinished signal."""
        self.roi.setPos([x, y],finish=False)
        self.roi.setSize([dx, dy],finish=False)


    def update(self, roi):
        """Update the second viewport which shows the selection / ROI"""
        self.img1b.setImage(self.roi.getArrayRegion(self.image, self.img1a),
                            levels=(0, self.image.max()))
        self.v1b.autoRange()
