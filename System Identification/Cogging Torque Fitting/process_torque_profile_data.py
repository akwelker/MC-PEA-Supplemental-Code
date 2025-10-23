'''
Adam Welker     Utah Robotics Center    2025

just a double check on the cogging torque fit -- 8/1/25
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from sys import path
path.append("..")

# Define Graph Colors
blue = '#1f77b4'
orange = '#ff7f0e'
green = '#2ca02c'
red = '#d62728'
purple = '#9467bd'
brown = '#8c564b'
pink = '#e377c2'
gray = '#7f7f7f'
yellow = '#bcbd22'

colors = [blue, orange, green, red, purple, brown, pink, gray, yellow]

Ts = 0.002 # Sampling time



### ------------------- Load Data ------------------- ###

# Load the data for feedback linearization steps
raw_data = pd.read_csv('cogging_Sweep_81.csv')
raw_data.columns = ['time', 'Motor Abs Position', 'Motor Abs Velocity', 'Accel 1', 'Accel 2', \
                            'Accel 3', 'Motor Inc Position', 'Motor Inc Velocity', 'Joint Position', 'Joint Velocity',
                            'Desired Torque', 'Desired Current', 'Actual Current', 'var 6', 'var 7', "Motor Temperature",
                            'var 8', 'Motor Voltage', 'var 9', 'var 10', 'var 11', 'test']


raw_data['time'] = raw_data['time'] * Ts # convert to seconds
raw_data['Joint Velocity'] = raw_data['Joint Velocity'] * 360 # convert to degrees per second
 

if False:
    plt.plot(raw_data['time'], raw_data['Joint Position'])
    plt.plot(raw_data['time'], raw_data['test'])
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (deg)')
    plt.title('Joint Position vs Time for 30 deg Step Test')
    plt.show()


test_data = raw_data[raw_data['test'] == 3]

times = test_data['time'].to_numpy()
joint_angle = test_data['Joint Position'].to_numpy()
joint_angle = np.radians(joint_angle)
joint_velocity = np.radians(test_data['Joint Velocity'].to_numpy())
torque = test_data['Actual Current'].to_numpy() * 0.098 * 6

for i in range(0,len(times)):

    if np.sign(joint_velocity[i]) != np.sign(torque[i]):

        joint_angle[i] = np.nan
        joint_velocity[i] = np.nan
        torque[i] = np.nan 

        
export_data = {'Time':times, 'Joint Angle':joint_angle, 'Motor Torque':torque}
new_data_frame = pd.DataFrame(export_data)

new_data_frame.to_csv('cogging_data_8125_UPDATE.csv')