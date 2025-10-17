'''
Adam Welker     Utah Robotics Center    2025

Impedance Modulation.py: Makes the impedance figure for the paper. that's all
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

Ts = 0.001 # Sampling time



### ------------------- Load Data ------------------- ###

# Load the data for feedback linearization steps
raw_data = pd.read_csv('impedance_data.csv')
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

# Set up the test parameters numbers 
test_numbers = [8,1,1,2,3]
ks = np.array([-6, -3, 0, 3, 6]) * 0.098/0.078 # Nm/rad

TORQUE = [0.750810886]

length_of_window = 500


# Make the segmented data arrays
real_k_values = []
kp_values = []
average_displacements = []
average_currents = []

times = []
joint_positions = []
currents = []


# Segment and analyze the data
for i in range(len(test_numbers)):
    
    # Fill out what we know
    test_num = test_numbers[i]
    k = ks[i]
    kp_values.append(k)
    real_k_values.append(k + 26.3) # 28.8 is the localized stiffness of the Cogging Torque Element

    # for test 2
    if i == 1:

        #load custom test data
        new_raw = pd.read_csv('impedance_data_2.csv')
        new_raw.columns = ['time', 'Motor Abs Position', 'Motor Abs Velocity', 'Accel 1', 'Accel 2', \
                            'Accel 3', 'Motor Inc Position', 'Motor Inc Velocity', 'Joint Position', 'Joint Velocity',
                            'Desired Torque', 'Desired Current', 'Actual Current', 'var 6', 'var 7', "Motor Temperature",
                            'var 8', 'Motor Voltage', 'var 9', 'var 10', 'var 11', 'test']
        
        new_raw['time'] = new_raw['time'] * Ts # convert to seconds
        new_raw['Joint Velocity'] = new_raw['Joint Velocity'] * 360 # convert to degrees per second
        # Segment by test number
        test_data = new_raw[new_raw['test'] == test_num]

        if True:
            plt.plot(new_raw['time'], new_raw['Joint Position'],'o')
            plt.plot(new_raw['time'], new_raw['test'],'o')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint Position (deg)')
            plt.title('Joint Position vs Time for 30 deg Step Test')
            plt.show()

    else:


        # Segment by test number
        test_data = raw_data[raw_data['test'] == test_num]

    time = test_data['time'].to_numpy()
    time = time - time[0]  # Start time at 0
    
    joint_position = test_data['Joint Position'].to_numpy()
    joint_velocity = test_data['Joint Velocity'].to_numpy()
    current = test_data['Actual Current'].to_numpy() * 0.098 * 6
    des_torques = test_data['Desired Torque'].to_numpy()


    # Now discern when the test starts
    start_index = 0

    for i in range(len(joint_position)):
        if joint_velocity[i] <= -1000:
            start_index = i
            break
    time = time[start_index:]
    joint_position = joint_position[start_index:]
    current = current[start_index:]
    joint_velocity = joint_velocity[start_index:]

    time = time[:1500]
    joint_position = joint_position[:1500]
    current = current[:1500]
    joint_velocity = joint_velocity[:1500]

    time = time - time[0]  # Start time at 0

    # Visual Check if needed
    if False:
        desired_time_start = None
        
        def on_click(event):
            global desired_time_start
            if event.xdata is not None and event.ydata is not None:
                desired_time_start = event.xdata
                print(f"Desired time start set to: {desired_time_start} seconds")
                plt.close()

        
        fig, ax = plt.subplots()
        ax.plot(time, joint_position, label=f'k = {k} Nm/rad')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Joint Position (deg)')
        ax.set_title('Joint Position vs Time for Stiffness Tests')
        ax.legend()
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        if desired_time_start is not None:
            # Find the index of the desired time start
            start_index = np.where(time >= desired_time_start)[0][0]
            time = time[start_index:1500]
            joint_position = joint_position[start_index:]
            current = current[start_index:]
            joint_velocity = joint_velocity[start_index:]

            time = time[:1500]
            joint_position = joint_position[:1500]
            current = current[:1500]
            joint_velocity = joint_velocity[:1500]

            time = time - time[0]  # Start time at 0

    # Another visual check if needed
    if False:
        plt.plot(time, joint_position, label=f'k = {k} Nm/rad')
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Position (deg)')
        plt.title('Joint Position vs Time for Stiffness Tests')
        plt.legend()
        plt.show()

    # Now find the average displacement and current

    average_displacement = np.mean(joint_position[-500:])
    average_current = np.mean(current[-500:])
    average_displacements.append(average_displacement)
    average_currents.append(average_current)

    # Now snip the segments to the length of the window
    time = time[:length_of_window]
    joint_position = joint_position[:length_of_window]
    current = current[:length_of_window]


    # Append the data to the lists
    times.append(time)
    joint_positions.append(joint_position)
    currents.append(current)

# Make the figure layout

fontsize = 9
plt.rcParams["font.size"] = fontsize


fig1 = plt.figure(figsize=(7.3,5), constrained_layout=True)

grid_spec = fig1.add_gridspec(2,1, height_ratios=[1, 0.75], wspace=0.005, hspace=0.01)

upper_grid = grid_spec[0].subgridspec(2, len(test_numbers), wspace=0.000, hspace=0.000)
lower_grid = grid_spec[1].subgridspec(1, 3, wspace=0.0, hspace=0.0)

# Populate the upper subgrid with the joint position and current plots

for i in range(len(test_numbers)):
    ax1 = fig1.add_subplot(upper_grid[0, i])
    ax1.margins(x=0)
    ax2 = fig1.add_subplot(upper_grid[1, i])

    # fig1.align_ylabels()

    # Plot the joint position
    ax1.plot(times[i], np.radians(joint_positions[i]), color=colors[i], label=f'$k_p$ = {kp_values[i]},K = {real_k_values[i]} (Nm/rad)', linewidth=2)
    ax1.set_title(f'$k_p$ = {kp_values[i]:.1f} Nm/rad\n $K_{{total}}$= {real_k_values[i]:.1f} Nm/rad', fontsize=fontsize)
    ax1.set_xlabel('Time (s)', fontsize=fontsize, labelpad=0)
    ax1.set_ylabel('$\\theta$ (rad)', fontsize=fontsize,labelpad=0)
    ax1.set_ylim(-.15, 0.1)
    plt.yticks([-np.pi/72,0,np.pi/72], fontsize=fontsize)
    ax1.set_xticks([0,0.5],['0','0.5'],fontsize=fontsize)


    # Plot the current
    ax2.plot(times[i], currents[i], color=colors[i], linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=fontsize)
    ax2.set_ylabel('$\\tau_m$ (Nm)', fontsize=fontsize,labelpad=0)
    plt.yticks([-1,0,1], fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    ax2.set_ylim(-1, 1)

    if i == 0:
        ax2.text(-0.25, -2., "A", fontsize=fontsize, fontweight='bold', color='black')

# Populate the lower subgrid with the average displacement and current plots
ax3 = fig1.add_subplot(lower_grid[0, 0])
ax4 = fig1.add_subplot(lower_grid[0, 1])


ax3.plot(ks, np.radians(average_displacements), 'k--', linewidth=1, alpha=0.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax4.plot(ks, average_currents, 'k--', linewidth=1, alpha=0.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
for i in range(len(test_numbers)):

    ax3.plot(ks[i], np.radians(average_displacements[i]), 'o', color=colors[i], markersize=6)
    ax4.plot(ks[i], average_currents[i], 'o', color=colors[i], markersize=6)

# Add in the lines for each plot
ax3.set_xlabel('Controller Stiffness (Nm/rad)', fontsize=fontsize)
ax3.set_ylabel('Steady-State $\\theta$ (rad)', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
ax3.set_xticks([-6,-3,0,3,6])
plt.yticks(fontsize=fontsize)
ax3.grid(True)
ax3.text(-11, -0.075, "B", fontsize=fontsize, fontweight='bold', color='black')

ax4.set_xlabel('Controller Stiffness (Nm/rad)', fontsize=fontsize)
ax4.set_ylabel('Steady-State $\\tau_m$ (Nm)', fontsize=fontsize)
plt.xticks([-6,-3,0,3,6],fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax4.grid(True)
ax4.text(-10,-.6, "C", fontsize=fontsize, fontweight='bold', color='black')

ax5 = fig1.add_subplot(lower_grid[0, 2])
ax5.plot(ks,np.array(average_currents)**2*.7*1.5, 'k--', linewidth=1, alpha=0.5)
for i in range(0,len(currents)):

    ax5.plot(ks[i], average_currents[i]**2*.7*1.5, 'o', color=colors[i], markersize=6)

ax5.grid()
ax5.set_ylabel('Joule Heating (W)')
ax5.set_xlabel('Controller Stiffness (Nm/rad)')
ax5.text(-10,-.05, "D", fontsize=fontsize, fontweight='bold', color='black')
ax5.set_xticks([-6,-3,0,3,6])



# Show the figure
plt.savefig('impedance_figure.pdf', dpi=300, bbox_inches='tight')
plt.show()
