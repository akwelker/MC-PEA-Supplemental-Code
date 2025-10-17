'''
Adam Welker         Utah Robotics       Summer 25

Torque_sensing.py: Using data collected on the steady-state
angle displacements of the MC-PEA while lifting various weights passively on 
different cogs, this program seeks to understand the variance
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from sys import path
path.append("..")


plt.rcParams['font.size'] = 9

# Define Graph Colors
blue = '#1f77b4'
orange = '#ff7f0e'
green = '#2ca02c'
red = '#d62728'
purple = '#9467bd'
brown = '#8c564b'
pink = '#e377c2'
gray = "#999898"
yellow = '#bcbd22'

colors = [blue, orange, green, red, purple, brown, pink, gray, yellow]

Ts = 0.001 # Sampling time


# Model of the magnetic cogging torque
def magnetic_torque(phi):

        a0=     0       
        a1=     0 
        b1=     2.4694
        a2=     0   
        b2=     -0.3620
        a3=     0 
        b3=     0.1477
        w=      12.0000 

        compensation_current = a0 + a1*np.cos(phi*w) + b1*np.sin(phi*w) + a2*np.cos(2*phi*w)\
                                + b2*np.sin(2*phi*w) + a3*np.cos(3*phi*w) + b3*np.sin(3*phi*w)

        return -compensation_current

# The actual torques hung from the the MC-PEA
REAL_TORQUE = [0.346162881,
                0.481045549,
                0.615928218,
                0.750810886,
                0.885693555,
                1.020576224,
                1.155458892,
                1.290341561,
                1.425224229,
                1.560106898,
                1.694989566,
                1.829872235
                ]

REAL_TORQUE = np.array(REAL_TORQUE)

stiction = np.abs(magnetic_torque(np.radians(0.5)))


MCT = 2.48

percent_MCT = REAL_TORQUE/MCT * 100

# Measured angles at cog 1 (0 deg)
cog1_data = np.array([[-0.424806,	-0.424806,	-0.380859,	-0.410156,	-0.395508],
                        [-0.629884,	-0.541992,	-0.454103,	-0.454103,	-0.454103],
                        [-1.06934,	-0.864259,	-1.01074,	-0.84961,	-0.966797],
                        [-1.17188,	-1.05469,	-1.06934,	-1.05469,	-1.20117],
                        [-1.52344,	-1.49414,	-1.46484,	-1.5234,	-1.55273],
                        [-1.88965,	-1.83106,	-1.91895,	-1.88965,	-1.96286],
                        [-2.22656,	-2.21191,	-2.19727,	-2.19727,	-2.27051],
                        [-2.47559,	-2.59277,	-2.51953,	-2.51953,	-2.49024],
                        [-2.79785,	-3.14941,	-2.90039,	-2.92969,	-2.79785],
                        [-3.47168,	-3.45703,	-3.42774,	-3.53027,	-3.57422],
                        [-3.63281,	-3.5887,	-3.5887,	-3.79395,	-4.13086],
                        [-4.2627,	-4.2332,	-4.40918,	-4.2627,	-4.21875]])

cog1_torques = magnetic_torque(np.radians(cog1_data)) + stiction
percent_errors_1 = np.zeros_like(cog1_torques)

for i in range(0,len(cog1_torques)):
        
        percent_errors_i = (REAL_TORQUE[i] - cog1_torques[i])/REAL_TORQUE[i]
        percent_errors_1[i] = percent_errors_i


#measured angles at cog 2 (30deg)
cog2_data = np.array([[-30.3369,	-30.3369,	-30.3516,	-30.3369,	-30.3223],
                        [-30.6152,	-30.7031,	-30.6592,	-30.835,	-30.7617],
                        [-31.0107,	-31.0107,	-30.9961,	-30.9961,	-31.0254],
                        [-31.1279,	-30.3037,	-30.2598,	-31.3916,	-31.1133],
                        [-31.5381,	-31.5674,	-31.5674,	-31.5234,	-31.5088],
                        [-31.5967,	-32.0801,	-32.0361,	-32.0361,	-32.0068],
                        [-32.2559,	-32.1973,	-32.2119,	-32.2119,	-32.2412],
                        [-32.4902,	-32.5049,	-32.4609,	-32.4463,	-32.4756],
                        [-32.8564,	-32.8125,	-32.9004,	-32.8125,	-32.8125],
                        [-33.5303,	-33.501,	-33.3398,	-33.4277,	-33.3984],
                        [-33.999,	-34.2041,	-33.9111,	-33.8818,	-33.8525],
                        [-34.2627,	-34.2627,	-34.2627,	-34.2188,	-34.2041]])



cog2_torques = magnetic_torque(np.radians(cog2_data)) + stiction
percent_errors_2 = np.zeros_like(cog2_torques)

for i in range(0,len(cog2_torques)):
        
        percent_errors_i = (REAL_TORQUE[i] - cog2_torques[i])/REAL_TORQUE[i]
        percent_errors_2[i] = percent_errors_i

#measured angles at cog 3
cog3_data = np.array([[-60.3223,	-60.3369,	-60.4541,	-60.3223,	-60.3516],
                        [-60.7764,	-60.7031,	-60.7617,	-60.8496,	-60.7324],
                        [-61.0107,	-61.0254,	-61.0107,	-61.0254,	-61.0254],
                        [-61.1133,	-61.1572,	-61.1426,	-61.1719,	-61.2451],
                        [-61.5234,	-61.4941,	-61.4795,	-61.5527,	-61.5234],
                        [-61.9629,	-61.9629,	-61.9336,	-61.9189,	-61.9336],
                        [-62.4023,	-62.2559,	-62.2119,	-62.2119,	-62.2119],
                        [-62.5342,	-62.6074,	-62.5635,	-62.5342,	-62.5342],
                        [-63.1934,	-63.2373,	-63.1641,	-63.1787,	-63.1787],
                        [-63.3691,	-63.5449,	-63.5742,	-63.6035,	-63.5156],
                        [-63.9111,	-63.7354,	-63.7354,	-64.0137,	-64.2627],
                        [-64.3066,	-64.0869,	-64.1602,	-64.2041,	-64.1309]])



cog3_torques = magnetic_torque(np.radians(cog3_data)) + stiction
percent_errors_3 = np.zeros_like(cog3_torques)

for i in range(0,len(cog3_torques)):
        
        percent_errors_i = (REAL_TORQUE[i] - cog3_torques[i])/REAL_TORQUE[i]
        percent_errors_3[i] = percent_errors_i



# Make the figure layout
fontsize = 9
plt.rcParams["font.size"] = fontsize
plt.figure(1,figsize=[5,2.5])
plt.subplot(121)
plt.plot(percent_MCT, REAL_TORQUE,'ko-', label="True Torque", markersize= 4)

for i in range(0,len(cog1_torques[0])):
        plt.plot(percent_MCT,cog1_torques[:,i],'o', color=colors[0], markersize=4)
        plt.plot(percent_MCT,cog2_torques[:,i],'o', color=colors[1], markersize=4)
        plt.plot(percent_MCT,cog3_torques[:,i],'o', color=colors[2], markersize=4)

plt.xlabel('% PCT')
plt.xlim([0,80])
plt.ylabel('Disturbance Torque (Nm)')
plt.ylim([0,2.4])
plt.legend(['True Torque', 'Est. Torque @ Cog 1', 'Est. Torque @ Cog 2', 'Est. Torque @ Cog 3'])
plt.grid()
plt.text(-5,-0.5,'A',weight="bold")

plt.subplot(122)

for i in range(0,len(cog1_torques[0])):
        plt.plot(percent_MCT,percent_errors_1[:,i],'o', color=colors[0])
        plt.plot(percent_MCT,percent_errors_2[:,i],'o', color=colors[1])
        plt.plot(percent_MCT,percent_errors_3[:,i],'o', color=colors[2])

plt.xlabel('% MCT')
plt.xlim([0,80])
plt.ylabel('Abs. % Error')
plt.ylim([0,40])
plt.grid()
plt.text(-5,-.2,'B', weight='bold')

plt.tight_layout()

# plt.savefig('propprioception.pdf',dpi=300)

# plt.show()


# Now check averages and stds

ave_torques = np.zeros(shape=(len(REAL_TORQUE), 3))
std_torques = np.zeros_like(ave_torques)

ave_p_errors = np.zeros_like(ave_torques)
std_p_errors = np.zeros_like(ave_torques)


for i in range(0,len(cog1_torques)):

        ave_torques[i,0] = np.median(cog1_torques[i])
        std_torques[i,0] = np.std(cog1_torques[i])
        ave_p_errors[i,0] = np.median(percent_errors_1[i])
        std_p_errors[i,0] = np.std(percent_errors_1[i])

        ave_torques[i,1] = np.median(cog2_torques[i])
        std_torques[i,1] = np.std(cog2_torques[i])
        ave_p_errors[i,1] = np.median(percent_errors_2[i])
        std_p_errors[i,1] = np.std(percent_errors_2[i])

        ave_torques[i,2] = np.median(cog3_torques[i])
        std_torques[i,2] = np.std(cog3_torques[i])
        ave_p_errors[i,2] = np.median(percent_errors_3[i])
        std_p_errors[i,2] = np.std(percent_errors_3[i])


# Make the figure layout
fontsize = 9
plt.rcParams["font.size"] = fontsize
plt.figure(2,figsize=[5,2.5])
plt.subplot(121)

plt.plot(percent_MCT, REAL_TORQUE,'ko-', label="True Torque", markersize= 4)
plt.plot(percent_MCT,ave_torques[:,0]-1000,'o-',markersize=4,color=gray, label='Estimated Torque')
plt.fill_between(percent_MCT,ave_torques[:,0]-1000+std_torques[:,0],
                        ave_torques[:,0]-1000 - std_torques[:,0], color=gray,
                         alpha=0.25, label = '$\pm \sigma$')
for i in range(0,len(ave_torques[0])):

        plt.plot(percent_MCT,ave_torques[:,i],'o-',markersize=4,color=colors[i])
        plt.fill_between(percent_MCT,ave_torques[:,i]+std_torques[:,i],
                         ave_torques[:,i] - std_torques[:,i], color=colors[i],
                         alpha=0.25)

plt.xlabel('% PCT')
plt.xlim([0,80])
plt.ylabel('Disturbance Torque (Nm)')
plt.ylim([0,2.0])
plt.legend(['True Torque', 'Est. Torque', '$\pm \sigma$'], loc='upper left')
plt.grid()
plt.text(-10,-0.35,'A',weight="bold")

plt.subplot(122)

lengends = ['RoA: $\\theta = 0$', 'RoA: $\\theta = \\frac{\pi}{6}$', 'RoA: $\\theta = \\frac{\pi}{3}$']
for i in range(0,len(ave_torques[0])):

        plt.plot(percent_MCT,-ave_p_errors[:,i],'o-',markersize=4,color=colors[i], label=lengends[i])
        plt.fill_between(percent_MCT,-ave_p_errors[:,i]+std_p_errors[:,i],
                         -ave_p_errors[:,i] - std_p_errors[:,i], color=colors[i],
                         alpha=0.25)
        
plt.fill_between(percent_MCT,ave_p_errors[:,i] -100+std_p_errors[:,i],
                        ave_p_errors[:,i] - 100 - std_p_errors[:,i], color=gray,
                        alpha=0.25, label='$\pm \sigma$')
plt.xlabel('% PCT')
plt.xlim([0,80])
plt.ylabel('Ratio of Error to True Torque')
plt.ylim([-.45,.45])
plt.grid()
plt.legend(loc = "lower right")
plt.text(-10,-.6,'B', weight='bold')

plt.subplots_adjust(top=0.983,
bottom=0.183,
left=0.115,
right=0.969,
hspace=0.2,
wspace=0.524)
plt.savefig('torque_sensing.svg',dpi=300)
plt.show()
plt.close()


'''
Now let's findout why there's 20% error

Could be a couple reasons
        - encoder resolution and bad numberical conditioning
        - belt stiffness
        - Stiffness of 3D printed parts

For each actual value, let's find the angle that we should be getting from the 
CTE using a newton's solver. Then we can find the difference in angle from
one to the next
'''

from scipy.optimize import fsolve

IDEAL_ANGLES= np.zeros((3,len(REAL_TORQUE)))

angle_errors_1 = np.zeros_like(cog1_data)
angle_errors_2 = np.zeros_like(cog2_data)
angle_errors_3 = np.zeros_like(cog3_data)

for i in range(0,len(REAL_TORQUE)):

        ideal_angle_1 = fsolve(lambda x: REAL_TORQUE[i]-.263 - magnetic_torque(x), np.radians(0))
        ideal_angle_2 = fsolve(lambda x: REAL_TORQUE[i]-.263 - magnetic_torque(x), np.radians(-30))
        ideal_angle_3 = fsolve(lambda x: REAL_TORQUE[i]-.263 - magnetic_torque(x), np.radians(-60))

        IDEAL_ANGLES[0,i] = ideal_angle_1
        IDEAL_ANGLES[1,i] = ideal_angle_2
        IDEAL_ANGLES[2,i] = ideal_angle_3

        for j in range(0,len(cog1_data[i])):

                angle_errors_1[i,j] = ideal_angle_1 - np.radians(cog1_data[i,j])
                angle_errors_2[i,j] = ideal_angle_2 - np.radians(cog2_data[i,j])
                angle_errors_3[i,j] = ideal_angle_3 - np.radians(cog3_data[i,j])



angle_error_mean_1 = np.degrees(np.mean(angle_errors_1, axis=1))
angle_error_mean_2 = np.degrees(np.mean(angle_errors_2, axis=1))
angle_error_mean_3 = np.degrees(np.mean(angle_errors_3, axis=1))

angle_error_median_1 = np.degrees(np.median(angle_errors_1, axis=1))
angle_error_median_2 = np.degrees(np.median(angle_errors_2, axis=1))
angle_error_median_3 = np.degrees(np.median(angle_errors_3, axis=1))

angle_error_std_1 = np.degrees(np.std(angle_errors_1, axis=1))
angle_error_std_2 = np.degrees(np.std(angle_errors_2, axis=1))
angle_error_std_3 = np.degrees(np.std(angle_errors_3, axis=1))


# print(np.degrees(IDEAL_ANGLES))

print('----------- Angle Error Means -----------')
print(angle_error_mean_1)
print(angle_error_mean_2)
print(angle_error_mean_3)

print('----------- Angle Error Medians -----------')
print(angle_error_median_1)
print(angle_error_median_2)
print(angle_error_median_3)

print('----------- Angle Error STDs -----------')
print(angle_error_std_1)
print(angle_error_std_2)
print(angle_error_std_3)



# Find how many encoder counts would create this error?

print('----------- Error in Terms of Encoder Counts ------------')
print(angle_error_mean_1/.33)
print(angle_error_mean_2/.33)
print(angle_error_mean_3/.33)


plt.figure(3)
plt.plot(percent_MCT, angle_error_mean_1/.33, '-o')
plt.plot(percent_MCT, angle_error_mean_2/.33, '-o')
plt.plot(percent_MCT, angle_error_mean_3/.33, '-o')
plt.title("error in terms of encoder count")



### Now look at the error in terms of a linear length
print('----------- Error in Terms of Arc Length (m) ------------')
print(np.radians(angle_error_mean_1)*96.5e-3/2)
print(np.radians(angle_error_mean_2)*96.5e-3/2)
print(np.radians(angle_error_mean_3)*96.5e-3/2)


plt.figure(4)
plt.plot(percent_MCT, np.radians(angle_error_mean_1)*96.5e-3/2, '-o')
plt.plot(percent_MCT, np.radians(angle_error_mean_2)*96.5e-3/2, '-o')
plt.plot(percent_MCT, np.radians(angle_error_mean_3)*96.5e-3/2, '-o')
plt.title("error in terms of arc length")
plt.show()


print('----------- Print Belt Deflections -----------------')
### Find the belt deflection
k_belt = 27000 * 0.25 * 4.44822 * (39.3701)
forces = -(REAL_TORQUE - .263)/(96.5e-3/2)

belt_defelctions = forces/k_belt

print(belt_defelctions)


arc_length_1 = np.radians(angle_error_mean_1)*96.5e-3/2
arc_length_2 = np.radians(angle_error_mean_2)*96.5e-3/2
arc_length_3 = np.radians(angle_error_mean_3)*96.5e-3/2


print('------------- Do the arc lengths match? ------------')
print(belt_defelctions- arc_length_1)
print(belt_defelctions- arc_length_2)
print(belt_defelctions- arc_length_3)

plt.figure(5)
plt.plot(percent_MCT, arc_length_1-belt_defelctions, '-o')
plt.plot(percent_MCT, arc_length_2-belt_defelctions, '-o')
plt.plot(percent_MCT, arc_length_3-belt_defelctions, '-o')
plt.title("Error including")
plt.show()

