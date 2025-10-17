'''
Adam Welker     Utah Robotics Center    Feb 2025

This script creates figures for step data taken on 2/4/25
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


plt.rcParams['font.size'] = 9
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

step_sizes = [180, 360, 720, 1080]

Ts = 0.002 # Sampling time


plt.rcParams['font.size'] = 9

### ------------------- Load Data ------------------- ###

# Load the data for feedback linearization steps
raw_data = pd.read_csv('continuous_rotation_data.csv')
raw_data.columns = ['time', 'Motor Abs Position', 'Motor Abs Velocity', 'Accel 1', 'Accel 2', \
                            'Accel 3', 'Motor Inc Position', 'Motor Inc Velocity', 'Joint Position', 'Joint Velocity',
                            'Desired Torque', 'Desired Current', 'Actual Current', 'var 6', 'var 7', "Motor Temperature",
                            'var 8', 'Motor Voltage', 'var 9', 'var 10', 'var 11', 'test']


raw_data['time'] = raw_data['time'] * Ts # convert to seconds
raw_data['Joint Velocity'] = raw_data['Joint Velocity'] # convert to degrees per second
 

if True:
    # plt.plot(raw_data['time'], raw_data['Joint Position'])
    plt.plot(raw_data['time'], raw_data['test'])
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (deg)')
    plt.title('Joint Position vs Time for FL Step Test')
    plt.show()

### ------------------- Segment Data ------------------- ###


# Separate out the different tests
test_1 = raw_data[raw_data['test'] == 1]
test_2 = raw_data[raw_data['test'] == 2]
test_3 = raw_data[raw_data['test'] == 3]
test_4 = raw_data[raw_data['test'] == 4]


# For each test, start the test when the joint position is below 5, and 
# the joint torque is above 0.5 Nm
tests = [test_1,test_2,test_3,test_4]
names = ['$\omega = 30deg/s',
         '$\omega = 90deg/s',
         '$\omega = 360 deg/s',
        '$\omega =  720 deg/s']
colors = [blue, orange, green, red]

times = []
joint_positions = []
joint_vels = []
currents = []

fig1 = plt.figure(1, figsize=[5,7])

for i in range(0, len(tests)):
    test = tests[i]
    
    start_index = 0

    for j in range(0, len(test)):
        if test['Desired Torque'].iloc[j] > 0.1 and i > 0:
            start_index = j
            break
    
    # Test 1 exception
    if i == 0 or i == 2:
        for j in range(0, len(test)):
            if abs(test['Joint Velocity'].iloc[j]) > 10:
                start_index = j
                break

        # if i == 0:

        #     start_index = start_index - 30

    test = test.iloc[start_index:]
    
    time_i = test['time'].to_numpy() # store the time vector
    time_i = time_i - time_i[0] # start time at 0
    joint_position_i = test['Joint Position'].to_numpy() # store the joint position vector
    current_i = test['Actual Current'].to_numpy()
    joint_vel_i = test['Joint Velocity'].to_numpy()

    times.append(time_i)
    joint_positions.append(joint_position_i)
    joint_vels.append(joint_vel_i)
    currents.append(current_i)


# Plot the segmented data

if False:

    for i in range(0, len(tests)):
        plt.plot(times[i], joint_positions[i], label='Test ' + str(i))
    

    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (deg)')
    plt.title('Joint Position vs Time for FL Step Test')
    plt.legend()
    plt.show()


### ------------------- Find Average Step + STD ------------------- ###



period = 4.0
period_step = int(period / Ts)
T_final = 30

Num_steps = int(T_final / period)

def get_average_step(joint_positions):
    '''
    Assuming the fixed parameters above, this function will find the average
    step from the joint position data.

    Args: joint_positions - a numpy array of joint positions for each step

    Returns: ave_poses - a numpy array of the average joint positions
             std_poses - a numpy array of the standard deviation of the 
                         joint positions
    '''

    ave_poses = []
    std_poses = []

    # Make an array of steps
    steps =[]
    time = np.arange(0, period, Ts)

    for i in range(0, Num_steps):

        start_time = i * period
        end_time = (i + 1) * period

        start_index = int(start_time / Ts)
        end_index = int(end_time / Ts)

        step = joint_positions[start_index:end_index]

        if len(step) == period_step:
            steps.append(np.array(step))
    
    # Plot the steps
    if False:
        for i in range(0, len(steps)):
            plt.plot(time, steps[i], label='Step ' + str(i))

        plt.xlabel('Time (s)')
        plt.ylabel('Joint Position (deg)')
        plt.title('Joint Position vs Time for FL Step Test')
        plt.legend()
        plt.show()

    # Find the average and standard deviation of the steps
    if len(steps) > 0:
        ave_poses = np.mean(steps, axis=0)
        std_poses = np.std(steps, axis=0)

        return ave_poses, std_poses

    else:

        return np.zeros_like(time), np.zeros_like(time)
    

def draw_steps(joint_positions, ax, display_color, zero_out = False):

    ave_poses = []
    std_poses = []

    # Make an array of steps
    steps = []
    time = np.arange(0, period, Ts)

    for i in range(0, Num_steps):

        start_time = i * period
        end_time = (i + 1) * period

        start_index = int(start_time / Ts)
        end_index = int(end_time / Ts)

        step = joint_positions[start_index:end_index]

        if zero_out == True:

            step = step - step[0]

        if len(step) == period_step and (i == 1 or i == 1 or i == 1):
            ax.plot(time,step,color=display_color)
    
    # Plot the steps
    if False:
        for i in range(0, len(steps)):
            plt.plot(time, steps[i], label='Step ' + str(i))

        plt.xlabel('Time (s)')
        plt.ylabel('Joint Position (deg)')
        plt.title('Joint Position vs Time for FL Step Test')
        plt.legend()
        plt.show()

    # Find the average and standard deviation of the steps
    if len(steps) > 0:
        ave_poses = np.mean(steps, axis=0)
        std_poses = np.std(steps, axis=0)

        return ave_poses, std_poses

    else:

        return np.zeros_like(time), np.zeros_like(time)



ave_times = []
ave_joint_poses = []
std_joint_poses = []
ave_joint_vels = []
std_joint_vels = []
ave_currents = []
std_currents = []


ax1 = fig1.add_subplot(411)
ax2 = fig1.add_subplot(412)
ax3 = fig1.add_subplot(413)
ax4  = fig1.add_subplot(414)

for i in range(len(tests)-1, -1, -1):

    # Add the average time. This is a vector of time step from 0 to 2 seconds
    ave_time = np.arange(0, period, Ts)
    ave_times.append(ave_time)

    ave_poses, std_poses = get_average_step(joint_positions[i])
    draw_steps(joint_positions[i],ax2,colors[i], zero_out=True)
    ave_joint_poses.append(ave_poses)
    std_joint_poses.append(std_poses)

    ave_vels, std_vels = get_average_step(joint_vels[i])
    draw_steps(joint_vels[i],ax1, colors[i])
    ave_joint_vels.append(ave_vels)
    std_joint_vels.append(std_vels)

    ave_current, std_current = get_average_step(currents[i])
    draw_steps(currents[i]*6*0.098,ax3,colors[i])
    ave_currents.append(ave_current)
    std_currents.append(std_current)

    ave_power, std_power = get_average_step(joint_vels[i]*currents[i]*6*0.098 + 1.5*.7*currents[i]**2)
    draw_steps(joint_vels[i]*currents[i]*6*0.098 + 1.5*.7*currents[i]**2,ax4,colors[i])



# Plot the average steps

def generate_ref_signal(time, step_size):
    '''
    Generate a reference signal for a step of a given size

    Args: time - a numpy array of time steps
          step_size - the size of the step

    Returns: ref_signal - a numpy array of the reference signal
    '''

    ref_signal = np.zeros_like(time)

    if len(time) == 1:
        if time[0] < period/2:
            return step_size
        else:
            return 0

    for i in range(1, len(time)):
        if time[i] < period/2:
            ref_signal[i] = step_size

    return ref_signal


# plt.figure(2, figsize=[5,5])
# plt.rcParams['font.size'] = 9

# for i in range(0,len(tests)):

#     ax1.plot(ave_times[i],ave_joint_vels[i], color=colors[i])
#     plt.plot(ave_times[i], generate_ref_signal(ave_times[i],step_sizes[i]), '--', color=colors[i])
#     plt.fill_between(ave_times[i], ave_joint_vels[i] + std_joint_vels[i], 
#                      ave_joint_vels[i] - std_joint_vels[i], color=colors[i],
#                      alpha = 0.5)
    

# plt.plot(ave_times[i], generate_ref_signal(ave_times[i],step_sizes[i]) - 1e6, '--', color='k', label='ref.')
# plt.plot(ave_times[i], ave_joint_vels[i]-1e6, color='k', label = '$\dot{\\theta}$')

# plt.fill_between(ave_times[i], 
#                     ave_joint_vels[i]-1e6 - std_joint_vels[i], 
#                     ave_joint_vels[i]-1e6 + std_joint_vels[i], 
#                     alpha=0.25, color='k',label='$\pm \sigma$')

# plt.legend(fontsize=9, loc='upper right')

# plt.xticks(fontsize=9)
# plt.yticks(fontsize=9)
# plt.xlim(-0.0625/2,2.5)
# plt.ylim(-720,1080)
# plt.yticks(np.arange(-720, 1080, 180), (str(item) for item in (np.arange(-720, 1080, 180).tolist())))
# plt.grid()
# plt.xlabel('Time (s)')
# plt.ylabel('Joint Velocity (Deg./s)')
# plt.savefig('Velocities.pdf')



for i in range(0,len(tests)):

    ax1.plot(ave_times[i], generate_ref_signal(ave_times[i],step_sizes[i]), '--', color=colors[i])

# plt.legend(fontsize=9, loc='upper right')

ax1.set_xlim(-0.0625/2,2.75)
ax1.set_ylim(-540,1440)
ax1.set_yticks(np.arange(-540, 1440, 360), (str(item) for item in (np.arange(-540, 1440, 360).tolist())))
ax1.grid()
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Joint Velocity (deg/s)')
ax1.plot(ave_times[i], generate_ref_signal(ave_times[i],step_sizes[i]) - 1e6, '--', color='k', label='ref.')
ax1.plot(ave_times[i], ave_joint_vels[i]-1e6, color='k', label = '$\omega$')
# ax1.legend()
ax1.text(-0.4,-1040,'A',weight='bold')
# fig1.savefig('Velocities_rawdata.pdf', dpi=300, bbox_inches=0)


ax2.set_xlim(-0.0625/2,2.75)
ax2.set_ylim(-180,2800)
ax2.set_yticks(np.arange(0, 2800, 360), (str(item) for item in (np.arange(0, 2800, 360).tolist())))
ax2.grid()
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Joint Angle (deg)')
ax2.plot(ave_times[i], ave_joint_vels[i]-1e6, color='k', label = '$\\theta$')
# ax2.legend()
ax2.text(-0.4,-720,'B',weight='bold')
# fig1.savefig('Velocities_rawdata.pdf', dpi=300, bbox_inches=0)



ax3.set_xlim(-0.0625/2,2.75)
ax3.set_ylim(-7,7)
# ax3.set_yticks(np.arange(0, 1600, 360), (str(item) for item in (np.arange(0, 1600, 360).tolist())))
ax3.grid()
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Motor Torque (Nm)')
ax3.plot(ave_times[i], ave_joint_vels[i]-1e6, color='k', label = '$\\tau_{m}$')
# ax3.legend()
ax3.text(-0.4,-10,'C',weight='bold')


ax4.set_xlim(-0.0625/2,2.75)
# ax4.set_ylim(-7,7)
# ax3.set_yticks(np.arange(0, 1600, 360), (str(item) for item in (np.arange(0, 1600, 360).tolist())))
ax4.grid()
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Power Consumption (W)')
# ax4.plot(ave_times[i], ave_joint_vels[i]-1e6, color='k', label = '$\\tau_{m}$')
# ax3.legend()
ax4.text(-0.4,-7000,'D',weight='bold')



fig1.tight_layout()
fig1.align_ylabels()
fig1.savefig('Velocity_Control.pdf', dpi=300, bbox_inches=0)


plt.show()