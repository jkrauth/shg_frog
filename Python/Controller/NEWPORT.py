"""
Driver for the Newport motorized translation stages connected  
to the SMC100 controller.                               
Commands and settings for serial communication are found in 
the SMC100 Newport Manual                     


File name: NEWPORT.py
Author: Julian Krauth
Date created: 2019/05/22
Python Version: 3.7                     
"""


#from .decorators import boundaries

import visa
from time import sleep


class SMC100:

    DEFAULTS = {'write_termination': '\r\n',
                'read_termination': '\r\n',
                'encoding': 'ascii',
                'baudrate': 921600,
                'timeout': 100,
                'parity': visa.constants.Parity.none,
                'data_bits': 8,
                'stop_bits': visa.constants.StopBits.one,
                'flow_control': visa.constants.VI_ASRL_FLOW_XON_XOFF,
                'query_termination': '?',
    }

    device = None
    
    def __init__(self,port,dev_number):
        self.port = port # e.g.: '/dev/ttyUSB0'
        self.dev_number = dev_number # e.g.: 1

                
    def initialize(self):
        port = 'ASRL'+self.port+'::INSTR'
        rm = visa.ResourceManager('@py')
        #rm_list = rm.list_resources()
        self.device = rm.open_resource(port,
                                       timeout=self.DEFAULTS['timeout'],
                                       encoding=self.DEFAULTS['encoding'],
                                       parity=self.DEFAULTS['parity'],
                                       baud_rate=self.DEFAULTS['baudrate'],
                                       data_bits=self.DEFAULTS['data_bits'],
                                       stop_bits=self.DEFAULTS['stop_bits'],
                                       flow_control=self.DEFAULTS['flow_control'],
                                       write_termination=self.DEFAULTS['write_termination'],
                                       read_termination=self.DEFAULTS['read_termination'])
        sleep(0.5) # make sure connection is established before doing anything else

        print(f"Connected to Newport stage {self.dev_number}: {self.idn}")

        #err, ctrl = self.error_and_controller_status() # clears error buffer
        #print(err, ctrl)
        #print("Connected to Newport stage: %s".format(self.idn))
        
    def write(self, cmd):
        cmd = f"{self.dev_number}{cmd}"
        self.device.write(cmd)

    def query(self, cmd):
        cmd_complete = f"{self.dev_number}{cmd}" # Add device number to command
        
        respons = self.device.query(cmd_complete)        
        # response is build the following way:
        # device_num+cmd_return+answer | cmd_return never contains the question mark
        dev_number = eval(respons[0])
        cmd_return = respons[1:3]
        answer = respons[3:]
        
        # Check for device number and command
        if (dev_number == self.dev_number) and (cmd_return == cmd.split('?')[0]):
            return answer
        else:
            raise Exception("Response contains wrong device number or wrong command.")

    def close(self):
        if self.device is not None:
            self.device.before_close()
            self.device.close()
        else:
            print('Newport device is already closed')

    @property
    def idn(self):
        idn = self.query("ID{}".format(self.DEFAULTS['query_termination']))
        return idn
        
    def wait_move_finish(self,interval):
        """ Interval given in seconds """
        errors, status = self.error_and_controller_status()
        while (status == self.CTRL_STATUS['moving']):
            errors, status = self.error_and_controller_status()
            sleep(interval)
        print("Movement finished")
            

    def error_and_controller_status(self):
        """Returns positioner errors and controller status
        This method also clears the error buffer.
        """
        respons = self.query('TS')
        positioner_errors = respons[0:4]
        controller_state = int(respons[4:6],16) # conv hex string to int
        return positioner_errors, controller_state

    CTRL_STATUS = {'configuration': 0x14,
                  'moving': 0x28,
                  'ready from homing': 0x32,
                  'ready from moving': 0x33,
                  'ready from disable': 0x34,
                  'ready from jogging': 0x35,}
    
    def move_rel(self,distance):
        self.write(f'PR{distance}')

    @property
    def position(self):
        pos = self.query(f"PA{self.DEFAULTS['query_termination']}")
        return pos

    def goto(self, pos):
        self.write(f'PA{pos}')        

    def home(self):
        self.write('OR')

    def reset(self):
        """After execution controller is in NOT REFERENCED state"""
        self.write('RS')

       
    @property
    def speed(self):
        speed = self.query(f"VA{self.DEFAULTS['query_termination']}")
        return speed

    @speed.setter
    def speed(self, value):
        self.write(f'VA{value}')

    @property
    def acceleration(self):
        accel = self.query(f"AC{self.DEFAULTS['query_termination']}")
        return accel

    @acceleration.setter
    def acceleration(self, value):
        self.write(f'AC{value}')

class SMC100DUMMY:
    """For testing purpose only"""

    device = None
    
    def __init__(self,port,dev_number):
        self.dev_number = dev_number
        self.idn = '123456'
        self.pos = 0
    
    def initialize(self):
        self.device = 1
        print(f"Connected to dummy Newport stage {self.dev_number}: {self.idn}")

    def write(self, cmd):
        pass

    def query(self, cmd):
        return 1

    def goto(self, pos):
        self.pos = pos

    @property
    def position(self):
        return self.pos

    def close(self):
        if self.device is not None:
            pass
        else:
            print('Newport device is already closed')

    def wait_move_finish(self,interval):
        pass

    

if __name__ == "__main__":

    print("This is the Conroller Driver example for the Newport Positioner.")
    port = '/dev/ttyUSB0'
    dev_id = 1
    dev = SMC100(port, dev_id)
    dev.initialize()
    #Do commands here
    dev.close()
