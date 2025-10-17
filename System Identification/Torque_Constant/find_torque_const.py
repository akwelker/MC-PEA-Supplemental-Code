'''
Adam Welker     Utah Robotics Center    Feb 2025

This script creates figures for step data taken on 2/4/25
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from scipy.optimize import curve_fit


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


inner_color = '#DDF2FF'
outer_color = '#A3D6F5'

colors = [blue, orange, green, red, purple, brown, pink, gray, yellow]

step_sizes = [180, 360, 720, 1080]

Ts = 0.002 # Sampling time


plt.rcParams['font.size'] = 9

### ------------------- Load Data ------------------- ###

# Load the data for feedback linearization steps
raw_data = pd.read_csv('torque_const4.csv')
raw_data.columns = ['time', 'Motor Abs Position', 'Motor Abs Velocity', 'Accel 1', 'Accel 2', \
                            'Accel 3', 'Motor Inc Position', 'Motor Inc Velocity', 'Joint Position', 'Joint Velocity',
                            'Desired Torque', 'Desired Current', 'Actual Current', 'var 6', 'var 7', "Motor Temperature",
                            'var 8', 'Motor Voltage', 'var 9', 'var 10', 'var 11', 'test']


raw_data['time'] = raw_data['time'] * Ts # convert to seconds
raw_data['Joint Velocity'] = raw_data['Joint Velocity'] # convert to degrees per second
 

if False:
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
test_5 = raw_data[raw_data['test'] == 5]
test_6 = raw_data[raw_data['test'] == 6]
test_7 = raw_data[raw_data['test'] == 7]
test_8 = raw_data[raw_data['test'] == 8]


start_times = [513, 593, 672, 869, 1152, 1326]


# For each test, start the test when the joint position is below 5, and 
# the joint torque is above 0.5 Nm
tests = [test_1,test_2,test_3,test_4, test_5, test_6, test_7, test_8]

torques = [2.099637572, 1.829872235, 1.557948775, 1.290341561, 1.020576224, 0.750810886, 0.481045549, 0.211280212
]
currents = []

for i in range(0, len(tests)):
    test = tests[i]

    # plt.plot(test['time'], test['Actual Current'])
    # plt.show()
    
    # test 1 exception
    

    # Now let's get an average of the currents
    current = np.abs(np.mean(test['Desired Current'].to_numpy()[0:]))
    currents.append(current)


###################### From Stack Overflow ###################
def f(x, A): # this is your 'straight line' y=f(x)
    return A*x

popt, pcov = curve_fit(f, currents,np.array(torques[0:len(currents)])) # your data x, y to fit
#############################################################

A = popt[0]
# b = popt[1]
b=0

print(f'A = {A}')
print(f'b = {b}')

solution_domain = np.linspace(0,3,101)
solution_range = solution_domain*A +b

SST = 0
SSE = 0
mean_torque = np.mean(torques)

for i in range(0,len(currents)):

    y = torques[i]
    y_hat = currents[i] * A

    SSE += (y-y_hat)**2
    SST += (y-mean_torque)**2


r_squared = 1 - SSE/SST


plt.figure(figsize=[2.3,2.3])
plt.rcParams['font.size'] = 9
plt.plot(solution_domain,solution_range, '-', color = 'k', label='$k_\\tau$ Model')
plt.plot(currents, np.array(torques[0:len(currents)]),'o', markersize=3,
         label="Collected Data",markeredgecolor='k',markerfacecolor=inner_color,
         markeredgewidth=0.25)

# plt.text(1.0, 1.5*(1.5*A+b), f'$\\tau$ = {A:.3f} $i$, $r^2 = {r_squared:.3f}$')
plt.xlabel('Motor Current (A)')
plt.ylabel('Motor Torque (Nm)')
plt.xlim([0,4])
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig('motor_torque_const.svg', transparent=True)
plt.show()

#Consider only output noise

def OLS(X_m, Y_m):
    '''
    Returns the Ordinary Least Squares Solution

    Args: X_m: (nx1) numpy arrary with measured input values
          Y_m: (nx1) numpy arrary with measured output values

    Returns: a: (float) the measured slope
    '''

    # assertions to make sure that sizes are correct
    try:
        assert X_m.shape[1] == 1
        assert Y_m.shape == X_m.shape
    except:

        print("OLS ERROR: X_m and Y_m are incorrect dimensions")
        return np.nan
    
    try:

        phi = np.hstack([X_m])
        a = np.linalg.solve(phi.T@phi, phi.T@Y_m)

        return 1/a.item(0)
    
    except:
        print("OLS ERROR: Solver did not return solution")
        return np.nan
    

tau_vector = np.array([torques]).T
curr_vectr = np.array([currents]).T

slope = OLS(tau_vector, curr_vectr)

print(f'OLS found: {slope}')