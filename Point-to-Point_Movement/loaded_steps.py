'''
Adam Welker     Utah Robotics Center    Feb 2025

This script creates figures for step data taken on 3/12/25
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from sys import path
path.append("..")

# Define Graph Colors
blue = "#1f77b4"
orange = '#ff7f0e'
green = '#2ca02c'
red = '#d62728'
purple = '#9467bd'
brown = '#8c564b'
pink = '#e377c2'
gray = '#7f7f7f'
yellow = '#bcbd22'

colors = [blue, orange, green, red, purple, brown, pink, gray, yellow]

Ts = 0.001 # Sampling time


test_numbers = [5,2,1] # The test numbers for the 30 degree step tests
filenames = ['loaded_steps(1).csv', 'loaded_steps(2).csv', 'loaded_steps(3).csv']
torques = [0.750810886, 1.290341561, 1.829872235, 2.369402909]



percent_cogging_torques = np.array(torques)/2.48*100

times= []
joint_positions = []
currents = []
torques = []
velocities = []


for i in range(len(test_numbers)):

    # Load the data for feedback linearization steps
    raw_data = pd.read_csv(filenames[i])
    raw_data.columns = ['time', 'Motor Abs Position', 'Motor Abs Velocity', 'Accel 1', 'Accel 2', \
                                'Accel 3', 'Motor Inc Position', 'Motor Inc Velocity', 'Joint Position', 'Joint Velocity',
                                'Desired Torque', 'Desired Current', 'Actual Current', 'var 6', 'var 7', "Motor Temperature",
                                'var 8', 'Motor Voltage', 'var 9', 'var 10', 'var 11', 'test']


    raw_data['time'] = raw_data['time'] * Ts # convert to seconds
    raw_data['Joint Velocity'] = raw_data['Joint Velocity'] # convert to degrees per second
    

    if False:
        plt.plot(raw_data['time'], raw_data['Joint Position'])
        plt.plot(raw_data['time'], raw_data['test'],'o')
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Position (deg)')
        plt.title('Joint Position vs Time for 30 deg Step Test')
        plt.show()


    test_num = test_numbers[i]

    test_data = raw_data[raw_data['test'] == test_num]

    time = test_data['time'].to_numpy()
    joint_position = test_data['Joint Position'].to_numpy()
    current = test_data['Actual Current'].to_numpy()
    des_torques = test_data['Desired Torque'].to_numpy()
    velocity = test_data['Joint Velocity'].to_numpy()

    # Find the time when the step starts
    start_index = 0

    for i in range(len(joint_position)):
        if abs(des_torques[i]) >= 0.5:
            start_index = i
            break

    time = time[start_index:]
    joint_position = joint_position[start_index:]
    current = current[start_index:]
    des_torques = des_torques[start_index:]
    velocity = velocity[start_index:]


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
    currents.append(current)
    torques.append(des_torques)
    velocities.append(velocity)

    print(np.max(current))



# Plot the data

font_size = 9
plt.rcParams.update({'font.size': font_size})

fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(311)

for i in range(len(test_numbers)):

    ax.plot(times[i], joint_positions[i], label= str(round(percent_cogging_torques[i])) + '% of PCT', color=colors[i])


def get_desired_position(t):
    
    
    
    stair_step = int(((t) / 2.0))

    stair_step = stair_step * 30 + 30

    return stair_step

desired_positions = [get_desired_position(t) for t in times[0]]
ax.plot(times[0], desired_positions, label='Reference', color='black', linestyle='--')


ax.set_ylabel('$\\theta$ (deg)')
# ax.set_title('Joint Position')
ax.set_xlim([-0.25, 8])
ax.set_ylim([0, 130])
ax.legend(loc='upper left')
ax.grid(True)

# Make zoomed in view  of t = 4.5-5.5

x1 = 4.75
x2 = 5.25
y1 = 80
y2 = 92

aspect_ratio = (ax.get_ylim()[1] - ax.get_ylim()[0])/(ax.get_xlim()[1] - ax.get_xlim()[0])
box_ratio = (y2-y1)/(x2-x1)
inset_width = .2

axins = ax.inset_axes(
    [0.7, 0.15, inset_width, inset_width/aspect_ratio*box_ratio],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[]) # [left, bottom, width, height]

for i in range(len(test_numbers)):
    axins.plot(times[i], joint_positions[i], label= str(round(percent_cogging_torques[i])) + '% of PCT', color=colors[i])

axins.plot(times[0], desired_positions, label='Desired Position', color='black', linestyle='--')

ax.indicate_inset_zoom(axins, edgecolor="black")

ax.text(-1,-10, 'A', fontsize=9, fontweight='bold')

ax = fig.add_subplot(312)

for i in range(len(test_numbers)):

    cleaned_current = np.zeros_like(currents[i])

    for j in range(10, len(currents[i])-11):

        cleaned_current[j] = np.average(currents[i][j-10:j+10])

        
    ax.plot(times[i], cleaned_current*0.098*6, label= str(round(percent_cogging_torques[i])) + '% of PCT', color=colors[i])


max_current = np.ones_like(times[0]) * 6.5
# ax.plot(times[0] - 4, max_current, label='Max Current Continous Motor Current', color='black', linestyle='--')
ax.set_xlabel('Time (s)')
ax.set_ylabel('$\\tau_m$ (Nm)')
# ax.set_title('Desired Motor Torque')
ax.set_xlim([-0.25, 8])
ax.set_ylim([0, 4])
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
ax.legend(loc = 'upper left')
ax.grid(True)
ax.text(-1,-0.5, 'B', fontsize=9, fontweight='bold')
# fig.suptitle('MC-PEA performance while picking up various disturbance loads \n 30 Degree Step', fontsize=16)

ax=plt.subplot(313)
for i in range(0,len(test_numbers)):

    cleaned_current = np.zeros_like(currents[i])

    for j in range(10, len(currents[i])-11):

        cleaned_current[j] = np.average(currents[i][j-10:j+10])

    ax.plot(times[i],velocities[i]*cleaned_current*0.098*6 + cleaned_current**2*1.5*0.7,
            label= str(round(percent_cogging_torques[i])) + '% of PCT', color=colors[i])

ax.set_xlabel('Time (s)')
ax.set_ylabel('Power Consumption (W)')
# ax.set_title('Desired Motor Torque')
ax.set_xlim([-0.25, 8])
ax.set_ylim([-50,50])
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
ax.legend(loc = 'upper left')
ax.grid(True)

# Make zoomed in view  of t = 4.5-5.5

x1 = 4.75
x2 = 5.25
y1 = -3.5
y2 = 3

aspect_ratio = (ax.get_ylim()[1] - ax.get_ylim()[0])/(ax.get_xlim()[1] - ax.get_xlim()[0])
box_ratio = (y2-y1)/(x2-x1)
inset_width = .2

axins = ax.inset_axes(
    [0.55, 0.1, inset_width, inset_width*box_ratio/aspect_ratio],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[]) # [left, bottom, width, height]

for i in range(len(test_numbers)):

    cleaned_current = np.zeros_like(currents[i])

    for j in range(10, len(currents[i])-11):

        cleaned_current[j] = np.average(currents[i][j-10:j+10])

    axins.plot(times[i],velocities[i]*cleaned_current*0.098*6 + cleaned_current**2*1.5*0.7,
            label= str(round(percent_cogging_torques[i])) + '% of PCT', color=colors[i])

axins.plot(times[0], desired_positions, label='Desired Position', color='black', linestyle='--')

ax.indicate_inset_zoom(axins, edgecolor="black")

ax.text(-1,-60, 'C', fontsize=9, fontweight='bold')
fig.align_ylabels()
plt.savefig('on_off_step_stiff_fig.svg')
plt.show()



    