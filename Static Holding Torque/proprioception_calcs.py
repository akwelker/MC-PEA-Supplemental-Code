'''
Finding torque estimates based off of passive displacement in stiff_steps.py
'''

import numpy as np


torques = [0.750810886, 1.290341561, 1.829872235]

thetas = np.radians([88.16, 86.51, 85.2])

stiction = 0#.263 #Nm

def magnetic_compesator(phi):

        a0=     0       
        a1=     0 
        b1=     3.6623  
        a2=     0   
        b2=     -0.5273 
        a3=     0 
        b3=     0.1734 
        w=      12.0000 

        compensation_current = a0 + a1*np.cos(phi*w) + b1*np.sin(phi*w) + a2*np.cos(2*phi*w)\
                                + b2*np.sin(2*phi*w) + a3*np.cos(3*phi*w) + b3*np.sin(3*phi*w)

        return -0.078*6*compensation_current


measured_torques = magnetic_compesator(thetas) + .268

print(measured_torques)

percent_errors = (measured_torques - torques)/torques * 100

print(percent_errors)