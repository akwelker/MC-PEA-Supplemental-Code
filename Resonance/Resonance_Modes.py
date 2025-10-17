'''
Adam Welker     CEM     Spring 2025

Resonance Mode Finder -- Helps the user find resonance modes in the MC-PEA
'''


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from tqdm import tqdm
from matplotlib import cm, ticker
from scipy.integrate import quad
from scipy.optimize import minimize


GENERATE = True
SHOW = True
SAVE = True


J = 0.00023532123382242692
b = 0.001182920473970913
R = 202e-3
K_t = 0.078


n = 12
K_values = [1,2]
even_sample_k = [2,4]
odd_sample_k = [1,3,5]

even_hot_starts = [(.727,135),(1.25,120)]
odd_hot_starts = [(0.482,190),(0.886,200),(1.5,150)]

even_optimals = []
odd_optimals = []



Ac = 2.4

def magnetic_compensator(theta):
    
    compensation_current = 0


    a0=     0       
    a1=     0
    b1=     2.4694
    a2=     0 
    b2=     -0.3620
    a3=     0
    b3=     0.1477
    w=      12.0000 


    compensation_current = a0 + a1*np.cos(theta*w) + b1*np.sin(theta*w) + a2*np.cos(2*theta*w)\
                            + b2*np.sin(2*theta*w) + a3*np.cos(3*theta*w) + b3*np.sin(3*theta*w)

    return compensation_current

# Get Control Torque
def get_tau_p(A,w,t):

    M = 1/np.sqrt(J**2*w**4 + b**2*w**2)

    phi = -np.pi - np.arctan(-b/J/w)

    return A/M * np.sin(w*t - phi)

    


# The motor current as a funtion of theta
def get_tau_m(A,w,t)->float:

    theta = (A*np.sin(w*t) + K*np.pi/n)

    tau_m = magnetic_compensator(theta)

    return tau_m


def control_effort(t,A,omega)->float:
    '''
    Function that calculates the integrand of 

    '''
    tau_m = get_tau_m(A,omega,t)

    tau_p = get_tau_p(A, omega,t)

    integrand = (tau_m + tau_p)**2

    return integrand


def null_effort(t,A,omega)->float:
    '''
    Function that calculates the integrand of the null control effort
    '''
    tau_m = get_tau_m(A,omega,t)

    tau_p = get_tau_p(A, omega,t)

    integrand = tau_p**2
    return integrand


def objective_function(A,omega)->float:

    '''
    The objective function we're optimizing
    '''

    t_0 = 0
    t_f = 2*np.pi/omega
    f = lambda x: control_effort(x,A,omega)

    f_Aw,_ = quad(f,t_0,t_f)

    return f_Aw

def null_objective_function(A,omega)->float:
    '''
    The objective function we're optimizing
    '''

    t_0 = 0
    t_f = 2*np.pi/omega
    f = lambda x: null_effort(x,A,omega)

    f_Aw,_ = quad(f,t_0,t_f)

    return f_Aw


fig = plt.figure(figsize=(7.3, 7), constrained_layout=True)
widths = [3.6,3.7]
heights = [3.5,3.5,3.5]
spec = fig.add_gridspec(ncols=len(K_values), nrows=3, width_ratios=widths, height_ratios=heights)

for s in range(len(K_values)):

    #============================ Get the Data ============================#
    K = K_values[s]

    A_range = np.linspace(0, np.pi/2, 500)
    w_range = np.linspace(40,350, 120)


    AA,ww = np.meshgrid(A_range, w_range, indexing='ij')

    energy_consumption = np.empty_like(AA)
    energy_consumption[:] = 0

    damper_consumption = np.empty_like(AA)
    damper_consumption[:] = 0

    consumption_ratio = np.empty_like(AA)
    consumption_ratio[:] = 0


    if GENERATE:
        print("Generating Plot...")
        for i in tqdm(range(len(A_range))):

            for j in range(len(w_range)):

                A_i = AA[i][j]
                w_j = ww[i][j]

                eng_output = objective_function(A_i, w_j)
                null_output = null_objective_function(A_i, w_j)

                energy_consumption[i][j] = eng_output
                damper_consumption[i][j] = null_output
                if null_output != 0:
                    consumption_ratio[i][j] = eng_output/null_output
                else:
                    consumption_ratio[i][j] = np.inf

        print("Plot Generated")
        print("Saving Data...")

        np.save(f"resonance_data/real_energy_consumption_K_{K}.npy", energy_consumption)
        np.save(f"resonance_data/real_damper_consumption_K_{K}.npy", damper_consumption)
        np.save(f"resonance_data/real_consumption_ratio_K_{K}.npy", consumption_ratio)
    else:

        print("Loading Data...")
        energy_consumption = np.load(f"resonance_data/real_energy_consumption_K_{K}.npy")
        damper_consumption = np.load(f"resonance_data/real_damper_consumption_K_{K}.npy")
        consumption_ratio = np.load(f"resonance_data/real_consumption_ratio_K_{K}.npy")
        print("Data Loaded")


#============================ Convert Values to Joule Heating ==================#
    energy_consumption = R/K_t**2*energy_consumption
    damper_consumption = R/K_t**2*damper_consumption

#============================ Setup the figure ============================#
    title_fontsize = 9
    legend_fontsize = 9
    label_fontsize = 9

    ax = fig.add_subplot(3,len(K_values), s+1+len(K_values))

    # fig.suptitle(f"Comparison of Expended Control Effort in Oscillation", fontsize=title_fontsize)


    #============================= Plot the MC-PEA Control Effort ============================#
    cs = ax.imshow(energy_consumption.T,
                    cmap=cm.rainbow, origin='lower', extent=(A_range[0], A_range[-1], w_range[0], w_range[-1]),aspect='auto', norm=colors.LogNorm(vmin=0.1, vmax=100))
    
    if s == 0:
        ax.set_ylabel('Frequency of Oscillation (rad/s)', fontsize=label_fontsize)
    
    if K % 2 == 0:
        # ax.set_title("K = 0,2,4...\n", fontsize=title_fontsize)

        for j in even_sample_k:
            ax.plot(j*np.pi/n*np.ones_like(w_range), w_range, 'w--', alpha=1, linewidth=2)
            
        # Now find the optimal values given the hot starts

        for i in range(len(even_hot_starts)):

            optimal = minimize(lambda x: objective_function(x[0], x[1]), even_hot_starts[i], bounds=((0, np.pi/2), (20, 350)), method='L-BFGS-B')
            optimal_point = optimal.x

            even_optimals.append(optimal_point)
            # ax.plot(optimal_point[0], optimal_point[1], 'r*', markersize=20, label='Local Minimum', ls='')

        
    else:
        # ax.set_title("K = 1,3,5...\n", fontsize=title_fontsize)
        for j in odd_sample_k:
            ax.plot(j*np.pi/n*np.ones_like(w_range), w_range, 'w--', alpha=1, linewidth=2)


        for i in range(len(odd_hot_starts)):
            optimal = minimize(lambda x: objective_function(x[0], x[1]), odd_hot_starts[i], bounds=((0, np.pi/2), (20, 350)), method='L-BFGS-B')
            optimal_point = optimal.x

            odd_optimals.append(optimal_point)
            # ax.plot(optimal_point[0], optimal_point[1], 'r*', markersize=20, label='Local Minimum', ls='')
        

    if K < K_values[-1]:
        cbar = fig.colorbar(cs)
        cbar.remove()
        ax.text(-0.2, 5, 'B', fontsize=label_fontsize, color='black', weight='bold', ha='center', va='bottom')

    else:
        cbar = fig.colorbar(cs)
        cbar.set_label("Joule Heating, MC-PEA", fontsize=label_fontsize)
        


    #============================= Plot the Damper Control Effort ============================#
    ax = fig.add_subplot(3,len(K_values),s+1)
    cs = ax.imshow(damper_consumption.T,
                    cmap=cm.rainbow, origin='lower', extent=(A_range[0], A_range[-1], w_range[0], w_range[-1]), aspect='auto', norm=colors.LogNorm(vmin=0.1, vmax=100))

    if s == 0:

        ax.set_ylabel('Frequency of Oscillation (rad/s)', fontsize=label_fontsize)

    if K % 2 == 0:

        for j in even_sample_k:
            ax.plot(j*np.pi/n*np.ones_like(w_range), w_range, 'w--', alpha=1.0, linewidth=2)
            
            if j != 2:
                ax.text(j*np.pi/n, np.max(w_range), f'{int(j/2)}$\phi$', fontsize=11, color='black', ha='center', va='bottom')
            else:
                ax.text(j*np.pi/n, np.max(w_range), f'$\phi$', fontsize=11, color='black', ha='center', va='bottom')
        
    else:

        for j in odd_sample_k:
            ax.plot(j*np.pi/n*np.ones_like(w_range), w_range, 'w--', alpha=1.0, linewidth=2)
            if j != 1:
                ax.text(j*np.pi/n, np.max(w_range), f'$\\frac{{{j}}}{{2}} \phi$', fontsize=11, color='black', ha='center', va='bottom')
            else:
                ax.text(j*np.pi/n, np.max(w_range), f'$\\frac{{{j}}}{{2}} \phi$', fontsize=11, color='black', ha='center', va='bottom')
        

    if K < K_values[-1]:
        cbar = fig.colorbar(cs)
        cbar.remove()
        ax.text(-0.2, 5, 'A', fontsize=label_fontsize, color='black', weight='bold', ha='center', va='bottom')
    else:
        cbar = fig.colorbar(cs)
        cbar.set_label("Joule Heating, Motor", fontsize=label_fontsize)

    #============================= Plot the Consumption Ratio ============================#
    ax = fig.add_subplot(3,len(K_values),s+1+len(K_values)*2)
    cs = ax.imshow(consumption_ratio.T,
                    cmap=cm.rainbow, origin='lower', extent=(A_range[0], A_range[-1], w_range[0], w_range[-1]), aspect='auto', norm=colors.LogNorm(vmin=0.01, vmax=10))


    if K < K_values[-1]:
        pass
    ax.set_xlabel('Amplitude of Oscillation (rad)', fontsize=label_fontsize)

    if s == 0:
        ax.set_ylabel('Frequency of Oscillation (rad/s)', fontsize=label_fontsize)
    plt.xticks(rotation=45)

    if K % 2 == 0:
        
        for j in even_sample_k:
            ax.plot(j*np.pi/n*np.ones_like(w_range), w_range, 'w--', alpha=1.0, linewidth=2)


        # plot the optimal points
        for i in range(len(even_optimals)):
            optimal = even_optimals[i]
            # ax.plot(optimal[0], optimal[1], 'r*', markersize=20, label='Local Minimum', ls='')

            # find the ratio at that point
            try:
                ratio = objective_function(optimal[0], optimal[1]) / null_objective_function(optimal[0], optimal[1])
            except ZeroDivisionError:
                ratio = np.nan
            # # ratio = objective_function(optimal[0], optimal[1]) / null_objective_function(optimal[0], optimal[1])
            # ax.text(optimal[0]+0.05, optimal[1]-15,
            #          f'({optimal[0]:.2f}, {optimal[1]:.2f}, {ratio:.2f})', 
            #          fontsize=legend_fontsize, color='black', ha='center', 
            #          va='bottom', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

             # now use the optimal MC-PEA point as a hot start for the optimal ratio point
            new_optimal = minimize(lambda x: objective_function(x[0], x[1]) / null_objective_function(x[0], x[1]), optimal, bounds=((1e-6, np.pi/2), (20, 350)), method='L-BFGS-B')
            new_optimal_point = new_optimal.x
            ax.plot(new_optimal_point[0], new_optimal_point[1], 'g*', markersize=8, label='Optimal Ratio Point', ls='')
            new_ratio = objective_function(new_optimal_point[0], new_optimal_point[1]) / null_objective_function(new_optimal_point[0], new_optimal_point[1])
            ax.text(new_optimal_point[0], new_optimal_point[1]+20, 
                    f'({new_ratio:.2f})', 
                    fontsize=legend_fontsize, color='black', ha='center', 
                    va='bottom', bbox=dict(facecolor='white', alpha=0.75, edgecolor='black', pad=2))
    else:
        
        for j in odd_sample_k:
            ax.plot(j*np.pi/n*np.ones_like(w_range), w_range, 'w--', alpha=1.0, linewidth=2)

        # plot the optimal points
        for i in range(len(odd_optimals)):
            K_near = odd_sample_k[i]
            optimal = odd_optimals[i]
            # ax.plot(optimal[0], optimal[1], 'r*', markersize=20, label='Local Minimum', ls='')

            # find the ratio at that point
            try:
                ratio = objective_function(optimal[0], optimal[1]) / null_objective_function(optimal[0], optimal[1])
            except ZeroDivisionError:
                ratio = np.nan
            # ax.text(optimal[0]+0.05, optimal[1]-15, 
            #         f'({optimal[0]:.2f}, {optimal[1]:.2f}, {ratio:.2f})', 
            #         fontsize=legend_fontsize, color='black', ha='center', 
            #         va='bottom', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

            # now use the optimal MC-PEA point as a hot start for the optimal ratio point
            new_optimal = minimize(lambda x: objective_function(x[0], x[1]) / null_objective_function(x[0], x[1]), optimal, bounds=(((K_near)*np.pi/n, (1+K_near)*np.pi/n), (20, 350)), method='L-BFGS-B')
            new_optimal_point = new_optimal.x
            print(f"New Optimal Point: {new_optimal_point}")
            ax.plot(new_optimal_point[0], new_optimal_point[1], 'g*', markersize=8, label='Optimal Ratio Point', ls='')
            new_ratio = objective_function(new_optimal_point[0], new_optimal_point[1]) / null_objective_function(new_optimal_point[0], new_optimal_point[1])
            if  K_near == 5:

                ax.text(new_optimal_point[0]-0.08, new_optimal_point[1]+20,
                     f'({new_ratio:.2f})', 
                     fontsize=legend_fontsize, color='black', ha='center', va='bottom',
                     bbox=dict(facecolor='white', alpha=0.75, edgecolor='black', pad=2))

            else:
                ax.text(new_optimal_point[0], new_optimal_point[1]+20,
                     f'({new_ratio:.2f})', 
                     fontsize=legend_fontsize, color='black', ha='center', va='bottom',
                     bbox=dict(facecolor='white', alpha=0.75, edgecolor='black', pad=2))

    if K < K_values[-1]:
        cbar = fig.colorbar(cs)
        cbar.remove()
        ax.text(-0.2, 5, 'C', fontsize=label_fontsize, color='black', weight='bold', ha='center', va='bottom')
    else:
        cbar = fig.colorbar(cs)
        cbar.set_label("Ratio of Joule Heating", fontsize=label_fontsize)

    CS = ax.contour(AA, ww, consumption_ratio, levels=[1], colors='k', linewidths=1, linestyles='--')
    ax.clabel(CS, inline=True, fontsize=9, fmt='%.1f', colors='k', manual=False)
       



if SAVE:
    print("Saving Figure...")
    plt.savefig(f"figures/resonance_figure.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"figures/resonance_figure.pdf", dpi=300)

if SHOW:
    plt.show()