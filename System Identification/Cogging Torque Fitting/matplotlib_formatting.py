'''
Adam Welker     July 25

Recreates new_mag_fig.fig in matplotlib
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io


inner_color = '#DDF2FF'
outer_color = '#A3D6F5'

data = io.loadmat('fit_81.mat')

sci_font_size = 9
plt.figure(figsize=[2.3,2.3])
plt.subplots_adjust(top=0.945,
                    bottom=0.175,
                    left=0.235,
                    right=0.935,
                    hspace=0.2,
                    wspace=0.2)


plt.xlim([-0.5,0.5])
plt.ylim([-8,8])

plt.plot(data['new_angle_data'][0],-data['new_torque_data'][0],'o', markersize=3,
         label="$\\tau_{mag}$ (Nm)",markeredgecolor=outer_color,markerfacecolor=inner_color,
         markeredgewidth=0.25)
plt.plot(data['fit_x'].T[0],-data['fit_y'].T[0],'k',label="Function Fit", linewidth=1)
plt.plot(data['CI_X1'].T[0],-data['CI_Y1'].T[0],'--k',label="95% CI", linewidth=0.5, dashes=(10, 5))
plt.plot(data['CI_X2'].T[0],-data['CI_Y2'].T[0],'--k', linewidth=.5, dashes=(20, 5))


max_torque = np.max(np.abs(data['fit_y']))

print(f'Maximum Cogging Torque: {max_torque}')

plt.xticks(fontsize=sci_font_size)
plt.yticks(fontsize=sci_font_size)
plt.grid()
plt.xlabel('Motor Angle (rad)', fontsize=sci_font_size)
plt.ylabel('$\\tau_{mag}$ (Nm)', fontsize=sci_font_size)
plt.legend(loc='upper right',fontsize=sci_font_size)


plt.savefig('magnetic_cogging_torque.png', dpi=300, transparent=True)
