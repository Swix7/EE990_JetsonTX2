# -*- coding: utf-8 -*-

import threading as tr
import time
import os
from enum import Enum
import subprocess as sp
import nvidia_smi as nvsm

from tensorflow.python.client import device_lib



    
class Device(Enum):
    desktop = 0
    jetson = 1
    server = 2
    
class PowerRead:

    def __init__(self, device = Device.desktop.value):
        self.device = device
        
        self.h = []
        local_device_protos = device_lib.list_local_devices()
        #Determines number of GPUs
        self.num_gpu = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
       
        if device in (Device.desktop.value, Device.server.value):
            print('Power Thing Initialized')
            nvsm.nvmlInit()
            for i in range(self.num_gpu):
                self.h += [nvsm.nvmlDeviceGetHandleByIndex(i)]
        
        self.p = []
        self.power_running = []
        
        for i in range(self.num_gpu):
            self.power_running += [True]
            self.p += [tr.Thread(target = self.read_power, args = (i,))]
            self.p[i].setDaemon(True)
            self.p[i].start()
    
    def read_power(self, i):
        pw = 0 #used for storing gpu power      
        try:
            with open("Power_Measurements_{}.txt".format(i), "a+") as f:
                if self.device in (Device.desktop.value, Device.server.value):
                    while self.power_running[i]:
                        time.sleep(1.0/500.0)#Sampling Rate
                        # Power in mW
                        f.write(str(nvsm.nvmlDeviceGetPowerUsage(self.h[i]))+"\n")
#                        print("READ POWER: {}".format(pw))
              
    
                elif(self.device == Device.jetson.value):
                    while self.power_running:
                        time.sleep(1.0/500.0)#Sampling Rate
                        temp = sp.check_output('cat /sys/bus/i2c/drivers/ina3221x/'
                                               +'0-0041/iio_device/in_power0_input',
                                               shell = True)
                        pw = float(temp.decode('utf-8'))
                        f.write(str(pw)+"\n")
#                        print("READ POWER: {}".format(pw))
                
                f.close()
        
        except:
            import traceback
            traceback.print_exc()
            
        return
    
    def stop(self):
        for i in range(self.num_gpu):
            self.power_running[i] = False
            self.p[i].join()
        
        
        