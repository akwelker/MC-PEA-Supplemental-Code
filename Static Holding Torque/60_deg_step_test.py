'''
Adam Welker     Utah Robotics Center    Feb 2025

This script creates figures for step data taken on 3/12/25
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

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

step_sizes = [5, 30, 60, 90]

Ts = 0.001 # Sampling time



### ------------------- Load Data ------------------- ###

# Load the data for feedback linearization steps
raw_data = pd.read_csv('1-dof-experimental_data/lifting3_28_2.csv')
raw_data.columns = ['time', 'Motor Abs Position', 'Motor Abs Velocity', 'Accel 1', 'Accel 2', \
                            'Accel 3', 'Motor Inc Position', 'Motor Inc Velocity', 'Joint Position', 'Joint Velocity',
                            'Desired Torque', 'Desired Current', 'Actual Current', 'var 6', 'var 7', "Motor Temperature",
                            'var 8', 'Motor Voltage', 'var 9', 'var 10', 'var 11', 'test']


raw_data['time'] = raw_data['time'] * Ts # convert to seconds
raw_data['Joint Velocity'] = raw_data['Joint Velocity'] * 360 # convert to degrees per second
 

if True:
    plt.plot(raw_data['time'], raw_data['Joint Position'])
    plt.plot(raw_data['time'], raw_data['test'])
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (deg)')
    plt.title('Joint Position vs Time for 30 deg Step Test')
    plt.show()


test_numbers = [1,3,4,5]
torques = [0.750810886, 1.290341561, 1.829872235, 2.369402909]

percent_cogging_torques = np.array(torques)/2.4*100

times= []
joint_positions = []
currents = []


for test_num in test_numbers:

    test_data = raw_data[raw_data['test'] == test_num]

    time = test_data['time'].to_numpy()
    joint_position = test_data['Joint Position'].to_numpy()
    current = test_data['Actual Current'].to_numpy()
    des_current = test_data['Desired Torque'].to_numpy()
    des_torques = test_data['Desired Torque'].to_numpy()

    # Find the time when the step starts
    start_index = 0

    for i in range(len(joint_position)):
        if abs(current[i]) >= 0.5:
            start_index = i
            break

    time = time[start_index:]
    joint_position = joint_position[start_index:]
    current = current[start_index:]
    des_current = des_current[start_index:]

    time = time - time[0] # Start time at 0

    #clean out artifacts of coms errors

    big_jump = 25
    for i in range(len(joint_position) - 2):


        forward_jump = joint_position[i+1] - joint_position[i]

        if abs(forward_jump) > big_jump:

            print(f'MAKING A FIX on test {test_num}')
            joint_position[i+1] = np.nan



    times.append(time)
    joint_positions.append(joint_position)
    currents.append(np.abs(des_current))



# Plot the data

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(211)

for i in range(len(test_numbers)):

    ax.plot(times[i], joint_positions[i], label= str(round(percent_cogging_torques[i])) + '% Cogging Torque Disturbance', color=colors[i])


def get_desired_position(t):
    
    
    
    stair_step = int(((t) / 2.0))

    stair_step = stair_step * 60 + 60

    return stair_step

desired_positions = [get_desired_position(t) for t in times[0]]
ax.plot(times[0], desired_positions, label='Desired Position', color='black', linestyle='--')


ax.set_ylabel('Joint Position (deg)')
ax.set_title('Joint Position')
ax.set_xlim([-0.5, 8])
ax.set_ylim([0, 360])
ax.legend()
ax.grid(True)

ax = fig.add_subplot(212)

for i in range(len(test_numbers)):
    ax.plot(times[i], currents[i], label= str(round(percent_cogging_torques[i])) + '% Cogging Torque Disturbance', color=colors[i])


max_current = np.ones_like(times[0]) * 6.5 * 0.135
# ax.plot(times[0], max_current, label='Max Current Continous Motor Current', color='black', linestyle='--')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Commanded Torque (Nm)')
ax.set_title('Commanded Motor Torque')
ax.set_xlim([-0.5, 8])
ax.legend()
ax.grid(True)

fig.suptitle('MC-PEA performance while picking up various disturbance loads \n 60 Degree Step', fontsize=16)

plt.savefig('figures/60_step_fig.png')

plt.show()



    