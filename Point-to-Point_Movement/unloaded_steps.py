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
gray = "#b8b8b8"
yellow = '#bcbd22'

colors = [blue, orange, green, red, purple, brown, pink, gray, yellow]

step_sizes = [90, 60, 30, 5]
step_sizes.reverse()

Ts = 0.002 # Sampling time


### ------------------- Load Data ------------------- ###

# Load the data for feedback linearization steps
raw_data = pd.read_csv('unloaded_steps.csv')
raw_data.columns = ['time', 'Motor Abs Position', 'Motor Abs Velocity', 'Accel 1', 'Accel 2', \
                            'Accel 3', 'Motor Inc Position', 'Motor Inc Velocity', 'Joint Position', 'Joint Velocity',
                            'Desired Torque', 'Desired Current', 'Actual Current', 'var 6', 'var 7', "Motor Temperature",
                            'var 8', 'Motor Voltage', 'var 9', 'var 10', 'var 11', 'test']


raw_data['time'] = raw_data['time'] * Ts # convert to seconds
raw_data['Joint Velocity'] = raw_data['Joint Velocity']
 

if False:
    plt.plot(raw_data['time'], raw_data['Joint Position'])
    plt.plot(raw_data['time'], raw_data['test'])
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (deg)')
    plt.title('Joint Position vs Time for FL Step Test')
    plt.show()


### ------------------- Load Data ------------------- ###

# Data from micro stepping

# Load the data for feedback linearization steps
innerwell_data = pd.read_csv('unloaded_steps.csv')
innerwell_data.columns = ['time', 'Motor Abs Position', 'Motor Abs Velocity', 'Accel 1', 'Accel 2', \
                            'Accel 3', 'Motor Inc Position', 'Motor Inc Velocity', 'Joint Position', 'Joint Velocity',
                            'Desired Torque', 'Desired Current', 'Actual Current', 'var 6', 'var 7', "Motor Temperature",
                            'var 8', 'Motor Voltage', 'var 9', 'var 10', 'var 11', 'test']


innerwell_data['time'] = innerwell_data['time'] * Ts # convert to seconds
innerwell_data['Joint Velocity'] = innerwell_data['Joint Velocity'] # convert to degrees per second
 

if False:
    plt.plot(innerwell_data['time'], innerwell_data['Joint Position'])

    plt.plot(innerwell_data['time'], innerwell_data['test'])
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (deg)')
    plt.title('Joint Position vs Time for FL Step Test')
    plt.show()


### ------------------- Segment Data ------------------- ###


# Separate out the different tests
test_1 = raw_data[raw_data['test'] == 4]
test_2 = raw_data[raw_data['test'] == 3]
test_3 = raw_data[raw_data['test'] == 2]
test_4 = innerwell_data[innerwell_data['test'] == 1]

# For each test, start the test when the joint position is below 5, and 
# the joint torque is above 0.5 Nm

tests = [test_1, test_2, test_3, test_4]
tests.reverse()
names = ['$3\phi$',
         '$2\phi$',
         ' $\phi$ ',
         '$\phi /6$']
colors = [blue, orange, green, red]


names.reverse()

times = []
joint_positions = []
currents = []
velocities = []

for i in range(0, len(tests)):
    test = tests[i]
    
    start_index = 0

    for j in range(0, len(test)):
        if test['Desired Torque'].iloc[j] > 0.5 and i > 0:
            start_index = j
            break
    
    # Test 1 exception
    if i == 1 or i == 0:
        for j in range(0, len(test)):
            if abs(test['Desired Torque'].iloc[j]) > 0.5:
                start_index = j
                break

    test = test.iloc[start_index:]
    
    time_i = test['time'].to_numpy() # store the time vector
    time_i = time_i - time_i[0] # start time at 0
    joint_position_i = test['Joint Position'].to_numpy() # store the joint position vector
    current_i = test['Actual Current'].to_numpy()
    vel_i = test['Joint Velocity'].to_numpy()

    times.append(time_i)
    joint_positions.append(joint_position_i)
    currents.append(current_i)
    velocities.append(vel_i)



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



period = 1.0
period_step = int(period / Ts)
T_final = 60.

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



ave_times = []
ave_joint_positions = []
std_joint_positions = []
ave_currents = []
std_currents = []
ave_velocities = []
std_velocities = []

for i in range(0, len(tests)):

    # Add the average time. This is a vector of time step from 0 to 2 seconds
    ave_time = np.arange(0, period, Ts)
    ave_times.append(ave_time)

    ave_poses, std_poses = get_average_step(joint_positions[i])
    ave_joint_positions.append(ave_poses)
    std_joint_positions.append(std_poses)

    ave_current, std_current = get_average_step(currents[i])
    ave_currents.append(ave_current)
    std_currents.append(std_current)

    ave_vel,std_vel = get_average_step(velocities[i])
    ave_velocities.append(ave_vel)
    std_velocities.append(std_vel)



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

fig = plt.figure(figsize=[3.5,5.5])
# plt.subplots_adjust(top=0.975,
#                     bottom=0.09,
#                     left=0.165,
#                     right=0.965,
#                     hspace=0.2,
#                     wspace=0.2)

plt.subplot(311)
for i in range(0, len(tests)):
    plt.plot(ave_times[i], ave_joint_positions[i], color=colors[i])
    
    plt.fill_between(ave_times[i], 
                        ave_joint_positions[i] - std_joint_positions[i], 
                        ave_joint_positions[i] + std_joint_positions[i], 
                        alpha=0.25, color=colors[i])
    
    # Add a step reference function
    ref_signal = generate_ref_signal(ave_times[i], step_sizes[i])
    plt.plot(ave_times[i], ref_signal, '--', color=colors[i], alpha = 0.75)

    # Add a text label
    plt.text(0.25, generate_ref_signal([0.25],step_sizes[i]) + 2, names[i], fontsize=9,
                color='k')

plt.plot(ave_times[i], ave_joint_positions[i]-1000, color='k', label = '$\\theta$')

plt.fill_between(ave_times[i], 
                    ave_joint_positions[i]-1000 - std_joint_positions[i], 
                    ave_joint_positions[i]-1000 + std_joint_positions[i], 
                    alpha=0.25, color='k',label='$\pm \sigma$')

# Add a step reference function
ref_signal = generate_ref_signal(ave_times[i], step_sizes[i])
plt.plot(ave_times[i], ref_signal-1000, '--', color = 'k', alpha = 0.75, label='Ref')

plt.xlabel('Time (s)', fontsize=9)
plt.ylabel('$\\theta$ (deg)', fontsize=9)
plt.yticks(np.arange(0, 100, 15))
# plt.title('Step Performance of Benchtop MC-PEA\n with Varying Step Sizes', 
#           fontsize=12)
plt.legend(fontsize=9, loc='upper right')
plt.grid()
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlim(-0.0625/2,1)
plt.ylim(-3,100)
plt.text(-0.15,-10,'A',fontsize=9,weight="bold")

plt.subplot(312)
for i in range(0, len(tests)):
    plt.plot(ave_times[i][:495], ave_currents[i][:495]*0.098*6, 
                label=names[i], color=colors[i])
    
    # plt.fill_between(ave_times[i], 
    #                     ave_currents[i] - std_currents[i], 
    #                     ave_currents[i] + std_currents[i], 
    #                     alpha=0.25, color=colors[i])
    pass
plt.xlabel('Time (s)', fontsize=9)
plt.ylabel('$\\tau_m$ (Nm)', fontsize=9)
plt.legend(['$\phi/6$', '$\phi$', '$2\phi$', '$3\phi$'], fontsize=9, loc='upper right')
plt.grid()
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlim(-0.0625/2,1)
plt.text(-0.15,-11,'B',fontsize=9,weight="bold")
plt.tight_layout()
fig.align_ylabels()


plt.subplot(313)

for i in range(0, len(tests)):
    plt.plot(ave_times[i][:495], 
             np.radians(ave_velocities[i][:495])* ave_currents[i][:495]*0.098*6 \
             +ave_currents[i][:495]**2*1.5*0.7, 
                label=names[i], color=colors[i])

plt.xlabel('Time (s)', fontsize=9)
plt.ylabel('Power Consumption (W)', fontsize=9)
plt.legend(['$\phi/6$', '$\phi$', '$2\phi$', '$3\phi$'], fontsize=9, loc='upper right')
plt.grid()
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlim(-0.0625/2,1)
plt.text(-0.15,-100,'C',fontsize=9,weight="bold")
plt.tight_layout()
fig.align_ylabels()

plt.savefig('step_performance_stiff.pdf', dpi=300)



### ------------------- Metric Functions ------------------- ###
def calc_falling_overshoot(data):

    subsection = data[250:500]

    initial_value = np.mean(subsection[0]) # get the initial joint position
    final_value = np.mean(subsection[225:245]) # get the final joint position

    final_diff = np.abs(initial_value - final_value)

    min_value = np.min(subsection)
    min_diff = np.abs(initial_value - min_value)

    return (min_diff - final_diff)/final_diff * 100

     

def calc_rising_overshoot(data):

    subsection = data[0:250]

    initial_value = np.mean(subsection[0]) # get the initial joint position
    final_value = np.mean(subsection[225:245]) # get the final joint position
    final_diff = np.abs(final_value - initial_value)

    max_value = np.max(subsection)
    max_diff = np.abs(max_value - initial_value)

    return max(0, (max_diff - final_diff)/final_diff) * 100



def calc_falling_settling_time(data):

    subsection = data[250:500]

    initial_value = np.mean(data[225:4250]) # get the initial joint position
    final_value = np.mean(subsection[225:250]) # get the final joint position

    final_diff = np.abs(initial_value - final_value)

    # Count backwards from the 900th index and find when it passes 5% of the
    # final value on either side

    for i in range(250,-1,-1):

        curr_val = np.mean(subsection[i-5:i+5]) # get the current joint position as an average. The data noise is messing up the current calculation

        curr_diff = np.abs(initial_value - curr_val)

        if abs(curr_diff-final_diff) > 0.05*final_diff:

            return i/1000
        
    return 0

    

def calc_rising_settling_time(data):

    subsection = data[0:250]

    final_value = np.mean(subsection[225:250]) # get the final joint position

    # Count backwards from the 900th index and find when it passes 5% of the
    # final value on either side

    for i in range(240,-1,-1):

        curr_val = np.mean(subsection[i-5:i+5]) # get the current joint position as an average. The data noise is messing up the current calculation

        curr_diff = np.abs(curr_val - final_value)

        if abs(curr_diff) > 0.05*final_value:

            return i/1000

    return 0

### ------------------- Find % Error ------------------- ###

rising_percent_errors = []
expected_values = [90, 60, 30, 5]
expected_values.reverse()

for i in range(0, len(tests)):

    ave_poses = ave_joint_positions[i]
    percent_error = np.abs(np.mean(ave_poses[225:250]) - expected_values[i])/expected_values[i] * 100
    rising_percent_errors.append(percent_error)

print(f"Rising Percent Errors: {rising_percent_errors}")

falling_percent_errors = []

for i in range(0, len(tests)):

    ave_poses = ave_joint_positions[i]
    percent_error = -(np.mean(ave_poses[450:500]))/expected_values[i] * 100
    falling_percent_errors.append(percent_error)

print(f"Falling Percent Errors: {falling_percent_errors}")

ave_per_errors = []

for i in range(0, len(tests)):
    ave_per_errors.append((np.abs(rising_percent_errors[i]) + np.abs(falling_percent_errors[i]))/2)

print(f"Average Percent Errors: {ave_per_errors}")


### ------------------- Find Settling Time ------------------- ###
f_settling_times = []
r_settling_times = []
settling_times = []

for i in range(0, len(tests)):
    f_settling_times.append(calc_falling_settling_time(ave_joint_positions[i]))
    r_settling_times.append(calc_rising_settling_time(ave_joint_positions[i]))
    settling_times.append((f_settling_times[i] + r_settling_times[i])/2)

print(f"Falling Settling Times: {f_settling_times}")
print(f"Rising Settling Times: {r_settling_times}")
print(f"Average Settling Times: {settling_times}")

### ------------------- Find Overshoot ------------------- ###
f_overshoots = []
r_overshoots = []
overshoots = []

for i in range(0, len(tests)):
    f_overshoots.append(calc_falling_overshoot(ave_joint_positions[i]))
    r_overshoots.append(calc_rising_overshoot(ave_joint_positions[i]))
    overshoots.append((f_overshoots[i] + r_overshoots[i])/2)

print(f"Falling Overshoots: {f_overshoots}")
print(f"Rising Overshoots: {r_overshoots}")
print(f"Average Overshoots: {overshoots}")


### Make a figure comparing FL to catch and throw methods

plt.figure(2, figsize=[5,2.5])
plt.subplot(121)
plt.plot(times[1][1000:1250]-2,joint_positions[1][1000:1250],label='Feedback\nLinearization')

comparison_data = pd.read_csv('../1-dof-experimental_data/comparison.csv')
comparison_data.columns = ['time', 'Motor Abs Position', 'Motor Abs Velocity', 'Accel 1', 'Accel 2', \
                            'Accel 3', 'Motor Inc Position', 'Motor Inc Velocity', 'Joint Position', 'Joint Velocity',
                            'Desired Torque', 'Desired Current', 'Actual Current', 'var 6', 'var 7', "Motor Temperature",
                            'var 8', 'Motor Voltage', 'var 9', 'var 10', 'var 11', 'test']


comparison_data['time'] = comparison_data['time'] * Ts # convert to seconds
comparison_data['Joint Velocity'] = comparison_data['Joint Velocity'] * 360 # convert to degrees per second

if False:
    plt.plot(comparison_data['time'], comparison_data['Joint Position'])
    plt.plot(comparison_data['time'], comparison_data['test'])
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (deg)')
    plt.title('Joint Position vs Time for FL Step Test')
    plt.show()

comparison_data = comparison_data[comparison_data['test'] == 2]
 

comp_time = comparison_data['time'].to_numpy()
comp_joint = comparison_data['Joint Position'].to_numpy()
comp_torque = comparison_data['Actual Current'].to_numpy()*0.098*6

start_index = 0

for i in range(0,len(comp_joint)):

    if abs(comp_torque[i]) > 0.5:

        start_index = i
        break

comp_time = comp_time[i:] - comp_time[i]
comp_joint = comp_joint[i:]
comp_torque = comp_torque[i:]


plt.plot(comp_time[0:250], comp_joint[0:250],label='Open Loop\nControl')
plt.xlabel('Time (s)',fontsize=9)
plt.ylabel('$\\theta$ (deg)', fontsize=9)
plt.grid()
plt.legend()
# plt.text(-0.15,-10,'A',fontsize=9,weight="bold")

plt.subplot(122)
plt.grid()
plt.plot(times[1][1000:1250]-2,currents[1][1000:1250]*.078*6,label='Feedback\nLinearization')
plt.plot(comp_time[0:250], comp_torque[0:250], label='Open Loop\nControl')
plt.xlabel('Time (s)',fontsize=9)
plt.ylabel('$\\tau_m$ (Nm)', fontsize=9)
plt.legend()
# plt.text(-0.15,-15,'B',fontsize=9,weight="bold")
plt.tight_layout()

plt.savefig('burst_comparison.png', dpi=300)
plt.show()