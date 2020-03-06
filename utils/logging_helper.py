"""
Author: Liangliang ZHENGT
Time: 2020/02/27
file: logging_helper.py

logging helper function for logging epochs and loss when training and testing
"""
import numpy as np
import time


def logging_helper(model, logfile,loss, epochs = 0,delta_time = 0, mode='train'):
    if mode != 'train' and mode != 'test' and mode != 'config' and mode != 'dev':
        raise TypeError('mode should be train or test or config.')
    
    if mode == 'config':
        with open(logfile, "a") as savefile:
            savefile.write(model.config+'\n')
    elif mode == 'train':
        with open(logfile, "a") as savefile:
            savefile.write(str(time.strftime("%X %x %Z"))+'\n')
            savefile.write("Epoch:" +str(epochs + 1)+ "training error is "+ str(loss)+'\n')
            savefile.write("Epoch:" +str(epochs + 1)+ "training time "+ str(delta_time)+' s\n')
    elif mode == 'test':
        logfile = 'TEST_'+logfile
        with open(logfile, "a") as savefile:
            savefile.write(str(model.config)+'\n')
            savefile.write(str(time.strftime("%X %x %Z"))+'\n')
            savefile.write("testing erro is :"+str(loss)+'\n')



