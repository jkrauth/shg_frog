"""
Driver class for the Allied Vision CCD camera.
It uses the python wrapper modul 'pymba' available on Github:
https://github.com/morefigs/pymba
version: pymba 0.3.6
The C/C++ libraries are provided by Allied Vision under the
name 'Vimba'.
version: Vimba 3.0

The camera originally was set to
IP Config Mode: Auto IP config mode
Gateway: 192.168.2.254
IP: 192.168.2.21
Subnet Mask: 255.255.255.0
The IP of the camera can be set by the user (persistent mode). 
I set it to 10.0.0.41. So it is in our subnet. Gateway is 0.0.0.0

File name: ALLIED_VISION_CCD.py
Author: Julian Krauth
Date created: 2019/11/14
Python Version: 3.7  
"""
import os
import sys
import yaml

import numpy as np
import time

import pymba

class CCDcamera:
        
    camera = None
    
    def __init__(self, camera_id='DEV_000F314E1E59'):

        # Start the camera package
        self.vimba = pymba.Vimba()
        self.vimba.startup()
        
        # Find cameras
        camera_ids = self.vimba.camera_ids()
        print ("Available cameras : %s" % (camera_ids))
        # Find correct camera index
        self.camera_id = camera_id
        for i in range(len(camera_ids)) :
            if self.camera_id == camera_ids[i]:
                self.camera_index = i
                break        

    def initialize(self):
        """Establish connection to camera"""
        self.camera = self.vimba.camera(self.camera_index)
        self.camera.open()
        print(f"Connected to camera : {self.camera_id}")
 
        
    def close(self):
        if self.camera is not None:
            self.camera.close()
            self.vimba.shutdown()
            #save settings (not implemented)

    @property
    def modelName(self) -> str:
        name = self.camera.DeviceModelName
        return name

    @property
    def exposure(self) -> int:
        """Exposure in microseconds"""
        expos = self.camera.ExposureTimeAbs
        return expos*1e-6

    @exposure.setter
    def exposure(self,expos: int):
        self.camera.ExposureTimeAbs = expos*1e6

    @property
    def gain(self):
        # Best image quality is achieved with gain = 0
        gain = self.camera.Gain
        return gain

    @gain.setter
    def gain(self,gain):
        self.camera.Gain = gain
            
    @property
    def roi_x(self) -> int:
        return self.camera.OffsetX

    @roi_x.setter
    def roi_x(self, val: int):
        self.camera.OffsetX = val
        
    @property
    def roi_y(self) -> int:
        return self.camera.OffsetY

    @roi_y.setter
    def roi_y(self, val: int):
        self.camera.OffsetY = val

    @property
    def roi_dx(self) -> int:
        return self.camera.Width
    
    @roi_dx.setter
    def roi_dx(self, val: int):
        self.camera.Width = val

    @property
    def roi_dy(self) -> int:
        return self.camera.Height

    @roi_dy.setter
    def roi_dy(self, val: int):
        self.camera.Height = val

    @property
    def sensorSize(self):
        """Returns number of pixels in width and height of the sensor"""
        width = self.camera.SensorWidth
        height = self.camera.SensorHeight
        return width, height
                
    @property
    def acquisitionMode(self):
        return self.camera.AcquisitionMode
        
    @acquisitionMode.setter
    def acquisitionMode(self, mode):
        OPTIONS = {'SingleFrame', 'Continuous'}
        if mode in OPTIONS:
            self.camera.arm(mode)
        else:
            raise Exception(f"Value '{mode}' for acquisition mode is not valied")
                  
    def takeSingleImg(self):
        """
        Sets everything to create a single image, takes the image 
        and returns it.
        The argument adapts the method for the pixel format set in
        the camera. See pixFormat method.
        """
        self.camera.arm('SingleFrame')
        frame = self.camera.acquire_frame()
        image = frame.buffer_data_numpy()
        self.camera.disarm()
        return image

    
    def trigMode(self, mode=None):
        """Toggle Trigger Mode set by 1/0, respectively.
        Keyword Arguments:
            mode {int} -- possible values: 0, 1
        Returns:
            int -- 0 or 1, depending on trigger mode off or on
        """
        if mode==None:
            onoff = {'Off': 0, 'On': 1}
            mode = self.camera.TriggerMode
            return onoff[mode]
        else:
            onoff = ['Off', 'On']
            self.camera.TriggerMode = onoff[mode]

    def trigSource(self, source=None):
        """Get/Select trigger source keyword arguments:
            source {str} -- Source can be one of the following strings:
                            'Freerun', 'Line1', 'Line2', 'FixedRate', 'Software'
        Returns:
            str -- trigger source
        """
        if source==None:
            source = self.camera.TriggerSource
            return source
        else:
            self.camera.TriggerSource = source


    def pixFormat(self, pix=None):
        """Get/Select pixel format
        Keyword Arguments:
            pix {str} -- possible values: 'Mono8','Mono12','Mono12Packed' 
                         (default: {None})
        Returns:
            str -- Pixelformat
        """
        if pix==None:
            pix = self.camera.PixelFormat
            return pix
        else:
            self.camera.PixelFormat = pix


class CCDcameraDummy:

    camera = None

    def __init__(self, camera_id=1):
        self.height = 1216
        self.width = 1936
        self.exposure = 20
        self.gain = 1
        self.roi_x = 0
        self.roi_y = 0
        self.roi_dx = 100
        self.roi_dy = 100
        self.source = 'External'
        self.format = 'Mono8'

    def initialize(self):
        self.camera = 1
        print('Connected to camera dummy!')

    def close(self):
        if self.camera is not None:
            pass
        else:
            print('Camera dummy closed!')

    def trigSource(self, val=None):
        if val is None:
            return self.source
        else:
            self.source = val

    def pixFormat(self, val=None):
        if val is None:
            return self.format
        else:
            self.format = val
    
    def takeSingleImg(self):
        return np.random.rand(self.height, self.width)
    
    @property
    def sensorSize(self):
        return self.width, self.height

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def millis():
        return int(round(time.time()*1000))
    # print(millis())

    t_steps = 20
    img_xoffset = 0
    img_yoffset = 0
    img_width = 1936
    img_height = 1216

    ccd = CCDcamera()

    ccd.initialize()
    # print(ccd.exposure())

    ccd.roi_x = img_xoffset
    ccd.roi_y = img_yoffset
    ccd.roi_dx = img_width
    ccd.roi_dy = img_height

    trace = np.zeros((img_width,t_steps))
    for i in range(t_steps):
        while millis() % 36 != 0:
            time.sleep(0.0009)

        img = ccd.takeSingleImg()
        print("image %d" % i)
        # Project image onto a single axis
        img_proj = np.divide(np.sum(img,0),img_height)
        trace[:,i] = img_proj
    
    plt.figure()
    plt.imshow(trace)
    plt.show()

    ccd.close()
