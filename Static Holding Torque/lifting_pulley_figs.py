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
raw_data = pd.read_csv('1-dof-experimental_data/lifting2.csv')
raw_data.columns = ['time', 'Motor Abs Position', 'Motor Abs Velocity', 'Accel 1', 'Accel 2', \
                            'Accel 3', 'Motor Inc Position', 'Motor Inc Velocity', 'Joint Position', 'Joint Velocity',
                            'Desired Torque', 'Desired Current', 'Actual Current', 'var 6', 'var 7', "Motor Temperature",
                            'var 8', 'Motor Voltage', 'var 9', 'var 10', 'var 11', 'test']


raw_data['time'] = raw_data['time'] * Ts # convert to seconds
raw_data['Joint Velocity'] = raw_data['Joint Velocity'] * 360 # convert to degrees per second
 

if False:
    plt.plot(raw_data['time'], raw_data['Joint Position'])
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (deg)')
    plt.title('Joint Position vs Time for FL Step Test')
    plt.show()


# convert raw data to numpy

times = raw_data['time'].to_numpy()
joint_angles = raw_data['Joint Position'].to_numpy()
torques = raw_data['Desired Torque'].to_numpy()

# find when abs(torque is large to find initial time step

start_index = 0
for i in range(0,len(times)):

    if abs(torques[i]) >= 0.1:

        start_index = i
        break

# now cut out the first part of the data
times = times[start_index:]
joint_angles = joint_angles[start_index:]
torques = torques[start_index:]

times = times - times[0] # start time at 0

# Now calculate the desired torque

step_length = 1.5 # seconds
step_size = -30 # degrees
offset = -30 # degrees

def get_desired_position(t):
    

    stair_step = int((t / step_length))

    stair_step = stair_step * step_size + offset

    return stair_step

desired_position = np.zeros(len(times))

for i in range(0,len(times)):

    desired_position[i] = get_desired_position(times[i])

# Now plot the data

plt.plot(times, joint_angles, label='Joint Angle')
plt.plot(times, desired_position, label='Reference signal $r(t)$', 
         linestyle='--', color='black')
plt.xlabel('Time (s)')
plt.ylabel('Joint Angle (deg)')
plt.title('Stepping With Pulley Weight, $\\tau_{disturbance} = 0.235$ Nm')
plt.xlim([0, 6])
plt.ylim([-150, 0])
plt.legend()
plt.grid()
ax = plt.gca()

# Add a data tip at T = 3 seconds
plt.annotate('Actuator Picks Up Weight',
             xy=(3, -90), xycoords='data',
             xytext=(2.5, -100), textcoords='data',
             arrowprops=dict(facecolor='red', shrink=0.01),
             horizontalalignment='right', verticalalignment='top')


# zoom in around t = 3 seconds and put and put an inset zoomed axis in the figure
# axins = plt.axes([0.2, 0.2, 0.15, 0.25])
# axins.plot(times, joint_angles, label='Joint Angle')
# axins.plot(times, desired_position, label='Reference signal $r(t)$', 
#          linestyle='--', color='black')
# axins.set_xlim(2.5, 3.5)
# axins.set_ylim(-100, -50)
# plt.yticks(visible=False)
# plt.xticks(visible=False)
# plt.grid(visible=True)

# ax.indicate_inset_zoom(axins, edgecolor="black")
# axins.set_title('Actuator Picks Up Weight')


plt.savefig('lifting_pulley_1.png')
plt.show()


# Estimate static load

ss_error = 90 - 86.51

k_total = 7.02 + 3.12044

load = np.radians(ss_error)*k_total - 0.03

weight = load / 23.5e-3

print(weight)

#===============================================================================
#                           Full Test
#===============================================================================

dist_torques = [0.750810886, 1.020576224, 1.290341561, 
                1.560106898, 1.829872235, 2.099637572, 2.369402909]

joint_angles = []
currents = []
times = []
torques = []
offsets = []
ignorable_steps = []


####### Now do the same with the 0.76nm #############

# Load the data for feedback linearization steps
raw_data = pd.read_csv('1-dof-experimental_data/lifting3_21_1.csv')
raw_data.columns = ['time', 'Motor Abs Position', 'Motor Abs Velocity', 'Accel 1', 'Accel 2', \
                            'Accel 3', 'Motor Inc Position', 'Motor Inc Velocity', 'Joint Position', 'Joint Velocity',
                            'Desired Torque', 'Desired Current', 'Actual Current', 'var 6', 'var 7', "Motor Temperature",
                            'var 8', 'Motor Voltage', 'var 9', 'var 10', 'var 11', 'test']


raw_data['time'] = raw_data['time'] * Ts # convert to seconds
raw_data['Joint Velocity'] = raw_data['Joint Velocity'] * 360 # convert to degrees per second


# test1 when test == 1
test1_data = raw_data[raw_data['test'] == 1]
double_step = raw_data[raw_data['test'] == 2]


# Plot the double step data
if False:
    plt.plot(raw_data['time'], raw_data['Joint Position'])
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (deg)')
    plt.title('Joint Position vs Time for 0.47NM Test')
    plt.show()

# Quick plot the data from the double step test
if True:

    double_step_time = double_step['time'].to_numpy()
    double_step_joint = double_step['Joint Position'].to_numpy()
    double_step_current = double_step['Actual Current'].to_numpy()
    double_step_torque = double_step['Desired Torque'].to_numpy()

    # Define start times for when desired torque is applied

    start_index = 0
    for i in range(0,len(double_step_time)):

        if abs(double_step_torque[i]) >= 0.1:

            start_index = i
            break

    double_step_time = double_step_time[start_index:]
    double_step_joint = double_step_joint[start_index:]
    double_step_current = double_step_current[start_index:]
    double_step_torque = double_step_torque[start_index:]

    double_step_time = double_step_time - double_step_time[0] # start time at 0
    double_step_joint = double_step_joint - 60 # start angle at 0

    offset = 60

    def get_desired_position(t):

        stair_step = int((t / 1.5))

        stair_step = stair_step * 60 + offset

        return stair_step
    
    desired_position = np.zeros(len(double_step_time))

    for i in range(0,len(double_step_time)):
        desired_position[i] = get_desired_position(double_step_time[i])

    #Plot it joint position
    plt.plot(double_step_time, double_step_joint)
    plt.plot(double_step_time, desired_position, linestyle='--', color='black')
    plt.legend(['Joint Position', 'Reference Signal'])
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (deg)')
    plt.title('Picking Up a Disturbance Load Torque of 0.48Nm \n 60 Degree Step')
    plt.xlim([0, 8])
    plt.ylim([0, 400])
    plt.grid() 
    plt.savefig('lifting_pulley_double_step.png')

    

joint1 = test1_data['Joint Position'].to_numpy()
current1 = test1_data['Actual Current'].to_numpy()
time1 = test1_data['time'].to_numpy()
torque1 = test1_data['Desired Torque'].to_numpy()

# find the initial time step

start_index = 0
for i in range(0,len(time1)):
    if abs(torque1[i]) >= 0.1:
        start_index = i
        break

time1 = time1[start_index:]
joint1 = joint1[start_index:]
current1 = current1[start_index:]
torque1 = torque1[start_index:]
time1 = time1 - time1[0] # start time at 0
joint1 = joint1 - 60 # start angle at 0


joint_angles.append(joint1)
currents.append(current1)
times.append(time1)
torques.append(torque1)
offsets.append(60)
ignorable_steps.append(0)


####### Now do the same with the 0.76nm #############

# Load the data for feedback linearization steps
raw_data = pd.read_csv('1-dof-experimental_data/lifting3_21_2.csv')
raw_data.columns = ['time', 'Motor Abs Position', 'Motor Abs Velocity', 'Accel 1', 'Accel 2', \
                            'Accel 3', 'Motor Inc Position', 'Motor Inc Velocity', 'Joint Position', 'Joint Velocity',
                            'Desired Torque', 'Desired Current', 'Actual Current', 'var 6', 'var 7', "Motor Temperature",
                            'var 8', 'Motor Voltage', 'var 9', 'var 10', 'var 11', 'test']


raw_data['time'] = raw_data['time'] * Ts # convert to seconds
raw_data['Joint Velocity'] = raw_data['Joint Velocity'] * 360 # convert to degrees per second


test_numbers = [1, 2, 3, 5]

for test_number in test_numbers:

    test_n_data = raw_data[raw_data['test'] == test_number]

    joint_n = test_n_data['Joint Position'].to_numpy()
    current_n = test_n_data['Actual Current'].to_numpy()
    time_n = test_n_data['time'].to_numpy()
    torque_n = test_n_data['Desired Torque'].to_numpy()

    # find the initial time step

    start_index = 0
    for i in range(0,len(time_n)):
        if abs(torque_n[i]) >= 0.1:
            start_index = i
            break

    time_n = time_n[start_index:]
    joint_n = joint_n[start_index:]
    current_n = current_n[start_index:]
    torque_n = torque_n[start_index:]
    time_n = time_n - time_n[0] # start time at 0
    joint_n = joint_n - 60 # start angle at 0

    joint_angles.append(joint_n)
    currents.append(current_n)
    times.append(time_n)
    torques.append(torque_n)
    offsets.append(60)
    ignorable_steps.append(0)


# Plot the raw data
if False:
    plt.plot(raw_data['time'], raw_data['Joint Position'])
    plt.plot(raw_data['time'], raw_data['test'])
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (deg)')
    plt.title('Joint Position vs Time for 0.47NM Test')
    plt.grid()
    plt.show()


####### Now do the same the next csv #############

# Load the data for feedback linearization steps
raw_data = pd.read_csv('1-dof-experimental_data/lifting3_21_3.csv')
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
    plt.title('Joint Position vs Time for 0.47NM Test')
    plt.grid()
    plt.show()



# Load the data for feedback linearization steps
raw_data = pd.read_csv('1-dof-experimental_data/lifting3_21_4.csv')
raw_data.columns = ['time', 'Motor Abs Position', 'Motor Abs Velocity', 'Accel 1', 'Accel 2', \
                            'Accel 3', 'Motor Inc Position', 'Motor Inc Velocity', 'Joint Position', 'Joint Velocity',
                            'Desired Torque', 'Desired Current', 'Actual Current', 'var 6', 'var 7', "Motor Temperature",
                            'var 8', 'Motor Voltage', 'var 9', 'var 10', 'var 11', 'test']


raw_data['time'] = raw_data['time'] * Ts # convert to seconds
raw_data['Joint Velocity'] = raw_data['Joint Velocity'] * 360 # convert to degrees per second

test7_data = raw_data[raw_data['test'] == 6]

joint7 = test7_data['Joint Position'].to_numpy()
current7 = test7_data['Actual Current'].to_numpy()
time7 = test7_data['time'].to_numpy()
torque7 = test7_data['Desired Torque'].to_numpy()
offset7 = 30

# find the initial time step

start_index = 0
for i in range(0,len(time7)):
    if abs(torque7[i]) >= 0.1:
        start_index = i
        break

time7 = time7[start_index:]
joint7 = joint7[start_index:]
current7 = current7[start_index:]
torque7 = torque7[start_index:]
time7 = time7 - time7[0] # start time at 0
joint7 = joint7 -30 # start angle at 0

joint_angles.append(joint7)
currents.append(current7)
times.append(time7)
torques.append(torque7)
offsets.append(offset7)
ignorable_steps.append(1)
 

if False:
    plt.plot(raw_data['time'], raw_data['Joint Position'])
    plt.plot(raw_data['time'], raw_data['test'])
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (deg)')
    plt.title('Joint Position vs Time for 0.47NM Test')
    plt.grid()
    plt.show()


# Load the data for feedback linearization steps
raw_data = pd.read_csv('1-dof-experimental_data/lifting3_21_5.csv')
raw_data.columns = ['time', 'Motor Abs Position', 'Motor Abs Velocity', 'Accel 1', 'Accel 2', \
                            'Accel 3', 'Motor Inc Position', 'Motor Inc Velocity', 'Joint Position', 'Joint Velocity',
                            'Desired Torque', 'Desired Current', 'Actual Current', 'var 6', 'var 7', "Motor Temperature",
                            'var 8', 'Motor Voltage', 'var 9', 'var 10', 'var 11', 'test']


raw_data['time'] = raw_data['time'] * Ts # convert to seconds
raw_data['Joint Velocity'] = raw_data['Joint Velocity'] * 360 # convert to degrees per second

test8_data = raw_data[raw_data['test'] == 7]

joint8 = test8_data['Joint Position'].to_numpy()
current8 = test8_data['Actual Current'].to_numpy()
time8 = test8_data['time'].to_numpy()
torque8 = test8_data['Desired Torque'].to_numpy()
offset8 = 60

# find the initial time step

start_index = 0
for i in range(0,len(time8)):
    if abs(torque8[i]) >= 0.1:
        start_index = i
        break

time8 = time8[start_index:]
joint8 = joint8[start_index:]
current8 = current8[start_index:]
torque8 = torque8[start_index:]
time8 = time8 - time8[0] # start time at 0
joint8 = joint8 - 30 # start angle at 0


joint_angles.append(joint8)
currents.append(current8)
times.append(time8)
torques.append(torque8)
offsets.append(offset8)
ignorable_steps.append(1)
 

if False:
    plt.plot(raw_data['time'], raw_data['Joint Position'])
    plt.plot(raw_data['time'], raw_data['test'])
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (deg)')
    plt.title('Joint Position vs Time for 0.47NM Test')
    plt.grid()
    plt.show()


####### Now plot all the data on top of each other #############

plt.figure()

percent_cogging_torque = np.array(dist_torques) / 2.4 * 100

for i in range(0,len(joint_angles)):

    if i % 2 == 0:

        if ignorable_steps[i] == 1:

            plot_angles = joint_angles[i][times[i] > 1.5]
            plot_times = times[i][times[i] > 1.5]

            plot_times = plot_times - plot_times[0]
            plot_angles = plot_angles - 30

            plt.plot(plot_times, plot_angles, label= str(round(percent_cogging_torque[i])) + '% Cogging Torque Disturbance')

        else:
            plt.plot(times[i], joint_angles[i], label= str(round(percent_cogging_torque[i])) + '% Cogging Torque Disturbance')


ref_times = np.linspace(0, 10, 1000)

def get_ref_angle(t):
    
    stair_step = int((t / 1.5))

    stair_step = stair_step * 30 + 30

    return stair_step

ref_angles = np.zeros(len(ref_times))

for i in range(0,len(ref_times)):
    ref_angles[i] = get_ref_angle(ref_times[i])

plt.plot(ref_times, ref_angles, linestyle='--', color='black', label='Reference Signal')

plt.xlabel('Time (s)')
plt.ylabel('Joint Position (deg)')
plt.title('Picking Up Various Disturbance Load Torques \n 30 Degree Step')
plt.xlim([0, 6])
plt.ylim([0, 150])
plt.legend()
plt.grid() 
plt.savefig('lifting_pulley_full_test.png')
plt.show()

