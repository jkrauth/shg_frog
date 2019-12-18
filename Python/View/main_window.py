"""
This module loads the MainWindow of the GUI from a QTdesigner file
and connects all widgets to the methods of the devices.

An example of how to run the code is found at the end of this file.

File name: main_window.py
Author: Julian Krauth
Date created: 2019/12/02
Python Version: 3.7
"""


import sys
import os
import yaml

from PyQt5 import QtWidgets, uic, QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
config_dir = os.path.join(CUR_DIR, "..", "..", "Examples", "config")
#sys.path.append(CUR_DIR)


class MainWindow(QtWidgets.QMainWindow):
    """This is the main window of the GUI for the FROG interface.
    The window is designed with Qt Designer and loaded into this class.
    """

    DEFAULTS = {
        'dev': { 
            0: 'ALLIED VISION CCD',
            1: 'ANDO Spectrometer',
        },
        'btn_connect': {
            True: 'Disconnect',
            False: 'Connect',
        },
        'btn_color': {
            True: 'rgb(239, 41, 41)',
            False: 'rgb(138, 226, 52)',
        }
    }

    def __init__(self, frog=None, parent=None):
        super().__init__(parent)

        # The object which is connected to the window
        self.frog = frog

        # Loading the GUI created with QTdesigner
        gui_path = os.path.dirname(__file__)
        uic.loadUi(os.path.join(gui_path, 'GUI/main_window.ui'), self)

        self.btn_connect.toggled.connect(self.connect_action)

        # Create Parametertree from FrogParams class
        self.par_class = FrogParams()
        # Print changes of parameters throughout operation
        self.par_class.printParChanges()
        self.par = self.par_class.par
        # Create ParameterTree widget filled with above parameters
        self.parTree = ParameterTree()
        self.parTree.setParameters(self.par, showTop=False)
        self.gridLayout.addWidget(self.parTree,1,0,1,2)


    @QtCore.pyqtSlot(bool)
    def connect_action(self, checked):
        # Get dictionaries
        dev = self.DEFAULTS['dev']
        btn = self.DEFAULTS['btn_connect']
        col = self.DEFAULTS['btn_color']
        # Get dropdown position
        index = self.dropdown.currentIndex()
        # Do GUI actions
        self.dropdown.setEnabled(not checked)
        if index==0: self.btn_roi.setEnabled(checked)
        self.btn_connect.setText(btn[checked])
        self.btn_connect.setStyleSheet(f"background-color:{col[checked]}")
        # Open device and respective parameter branch
        if checked:            
            self.frog.initialize(index)
            self.par.param(dev[index]).show()
            self.par.param('Newport Stage').show()
        else:
            self.frog.close(index)
            self.par.param(dev[index]).hide()
            self.par.param('Newport Stage').hide()
        # needed for updating par tree in GUI
        self.parTree.setParameters(self.par, showTop=False)

        
    

class FrogParams:
    """
    Class which implements the parameters used by the parametertree widget.
    """
    # Load defaults:
    with open(config_dir + '/config.yml','r') as f:
        CONFIG = yaml.load(f,Loader=yaml.FullLoader)
    DEFAULTS = {'maxW': CONFIG['pxls width'],
                'maxH': CONFIG['pxls height'],
    }

    def __init__(self):
        # Define parameters for parametertree
        params = [
            {'name':'Phase Retrieval','type':'group','visible':True,
             'children': [
                 {'name':'prepFROG Size', 'type': 'int', 'value': 128},
                 {'name':'Seed', 'type': 'list', 'values': {"Random":0,"fromFile":1}, 'value':0},
                 {'name':'Max. Iterations', 'type': 'int', 'value': 200},
                 {'name':'G Tolerance', 'type': 'float', 'value': 0.001}
             ]},
            {'name':'ANDO Spectrometer','type':'group','visible':False,
             'children': [
                 {'name':'Center', 'type': 'float', 'value': 390.},
                 {'name':'Span', 'type': 'float', 'value': 20.},
                 {'name':'CW mode', 'type': 'bool', 'value': True},
                 {'name':'Rep. time', 'type': 'float', 'value': 36},
                 {'name':'Sampling', 'type': 'int', 'value': 201}
             ]},
            {'name':'ALLIED VISION CCD','type':'group','visible':False,
             'children': [
                 {'name':'Exposure', 'type': 'float', 'value': 0.036, 'dec': True, 'step': 1, 'siPrefix': True, 'suffix': 's'},
                 {'name':'Gain', 'type': 'float', 'value': 0, 'dec': True, 'step': 1},                 
                 {'name':'Crop Image', 'type': 'group', 'expanded':False,
                  'children': [
                     {'name':'Width','type':'int',
                      'value': 600,'limits':[1,self.DEFAULTS['maxW']],'step': 2},
                     {'name':'Height','type':'int',
                      'value': 10,'limits':[1,self.DEFAULTS['maxH']],'step': 2},
                     {'name':'Xpos','type':'int','value': 400,'step': 2},
                     {'name':'Ypos','type':'int','value': 470,'step': 2}
                 ]},
                 {'name':'Trigger', 'type': 'group', 'children': [
                     {'name':'Mode','type':'list','visible':False,
                      'values': {"On":1,"Off":0},'value':1},
                     {'name':'Source','type':'list','visible':True,
                      'values': {"Freerun":'Freerun',
                                 "External":'Line1'
                                 #'Line2':'Line2',
                                 #'FixedRate':'FixedRate',
                                 #'Software':'Software'
                      },'value':'External'}
                 ]},
                 {'name':'PixelFormat', 'type': 'list',
                  'values': {'Mono8':'Mono8','Mono12':'Mono12'
                             #'Mono12Packed':'Mono12Packed'
                  },
                  'value':'Mono8'},
             ]},
            {'name':'Newport Stage','type':'group','visible':False,
             'children': [
                 {'name':'Current Position','type':'float','value': 0.},
                 {'name':'Offset','type':'float','value': 11370,
                  'limits':[0,25000]},
                 {'name':'Start Position','type':'float','value': -256},
                 {'name':'Step Size','type':'float','value': 4.},
                 {'name':'Number of steps','type':'int','readonly':True,
                  'value': 128}
             ]}
        ]
        
        # Create parameter objects
        self.par = Parameter.create(name='params',
                              type='group',
                              children=params)
        
        ### Some settings regarding CCD parameters ###
        crop_par = self.par.param('ALLIED VISION CCD').child('Crop Image')
        width_par = crop_par.child('Width')
        height_par = crop_par.child('Height')
        xpos_par = crop_par.child('Xpos')
        ypos_par = crop_par.child('Ypos')
        maxW = self.DEFAULTS['maxW']
        maxH = self.DEFAULTS['maxH']
        width_par.setLimits([1,maxW-xpos_par.value()])
        height_par.setLimits([1,maxH-ypos_par.value()])
        xpos_par.setLimits([0,maxW-width_par.value()])
        ypos_par.setLimits([0,maxH-height_par.value()])
        crop_par.sigTreeStateChanged.connect(self.setCropLimits)
        
        ### Some settings regarding the Stage parameters ###
        stage_par = self.par.param('Newport Stage')
        start_par = stage_par.child('Start Position')
        step_par = stage_par.child('Step Size')
        off_par = stage_par.child('Offset')
        
        # Set limits of Start Position, depending on offset
        start_par.setLimits([-off_par.value(),-0.2])
        off_par.sigValueChanged.connect(self.setStartPosLimits)

        # Set limits of Step Size, depending on Start Position
        step_par.setLimits([0.2,abs(start_par.value())])
        start_par.sigValueChanged.connect(self.setStepLimits)
        
        # Always update Number of steps, given by start pos and step size
        start_par.sigValueChanged.connect(self.showSteps)
        step_par.sigValueChanged.connect(self.showSteps)

    #@pyqtSlot()
    def setCropLimits(self,param,changes):
        maxW = self.DEFAULTS['maxW']
        maxH = self.DEFAULTS['maxH']
        for param, change, data in changes:
            path = self.par.childPath(param)
            par = self.par.param(path[0]).child(path[1])
            if path[2]=='Width':
                mx = maxW
                par.child('Xpos').setLimits([0,mx-par.child(path[2]).value()])
            if path[2]=='Height':
                mx = maxH
                par.child('Ypos').setLimits([0,mx-par.child(path[2]).value()])
            if path[2]=='Xpos':
                mx = maxW
                par.child('Width').setLimits([1,mx-par.child(path[2]).value()])
            if path[2]=='Ypos':
                mx = maxH
                par.child('Height').setLimits([1,mx-par.child(path[2]).value()])
            
    
    #@pyqtSlot()
    def setStepLimits(self,param,val):
        step_par = self.par.param('Newport Stage').child('Step Size')
        step_par.setLimits([0.2,abs(val)])
        
    #@pyqtSlot()
    def setStartPosLimits(self,param,val):
        start_pos = self.par.param('Newport Stage').child('Start Position')
        start_pos.setLimits([-val,-0.2])
        
    #@pyqtSlot(float)
    def showPos(self,val):
        pos = self.par.param('Newport Stage').child('Current Position')
        pos.setValue(val)

    #@pyqtSlot()
    def showSteps(self,dummy):
        start_pos = self.par.param('Newport Stage').child('Start Position')
        step_size = self.par.param('Newport Stage').child('Step Size')
        val = int(round(2*abs(start_pos.value())/step_size.value()))

        num = self.par.param('Newport Stage').child('Number of steps')
        num.setValue(val)

    def printParChanges(self):
        # Do print changes in parametertree
        self.par.sigTreeStateChanged.connect(self._Change)
        
    def _Change(self, param, changes):
        ## If anything changes in the parametertree, print a message
        for param, change, data in changes:
            path = self.par.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()
            print("tree changes:")
            print('  parameter: %s'% childName)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')

    """ End FrogParams class """


if __name__ == "__main__":

    import sys
    

    app = QtGui.QApplication([])
    win = MainWindow()
