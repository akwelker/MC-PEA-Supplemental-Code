'''
Adam Welker     MEEN 7200   Fall 2024

PhasePortraitPlotter.py: This file contains the PhasePlotter class 
which can be used to make Phase Portraits of 2D and 3D systems of
ODEs.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class PhasePlotter:

    test_points = np.array([])
    eq_points = np.array([])

    lyaponov_candidate = None
    lyaponov_derivative_candidate = None

    def __init__(self, system_type:str):
        '''
        Constructor for the PhasePlotter class.

        Args:
            system_type (str): The type of system to be plotted. 
                This can be either '2D' or '3D'.

        Returns:

        '''

        if system_type not in ['2D', '3D']:
            
            raise ValueError('Error: System_type must be either 2D or 3D')
        
        else:
            self.system_type = system_type


    def set_test_points(self, points:np.ndarray):
        '''
        This method sets the test points for the phase portrait.

        Args:
            points (np.ndarray): An array of test points for the phase portrait.

        Returns:
        '''

        self.test_points = points

    def set_x_bar_points(self, points:np.ndarray):
        '''
        This method sets the equilibrium points for the phase portrait.

        Args:
            points (np.ndarray): An array of equilibrium points for the phase portrait.

        Returns:
        '''

        self.eq_points = points

    
    def set_system(self, system:callable):
        '''
        This method sets the system of ODEs to be plotted.

        Args:
            system (callable): A callable function that takes in the current state
            of the system and returns the derivative of the system.

        Returns:
        '''

        self.system = system

    def set_lyapunov_candidate(self, lyaponov_candidate:callable):
        '''
        This method sets the lyaponov candidate function for the system. This 
        function is added as a countour plot in the phase portrait.

        Args:
            lyaponov_candidate (callable): A callable function that takes in the 
            current state of the system  as an array x and 
            returns the lyaponov candidate function value at that point.

        Returns:
        '''

        self.lyaponov_candidate = lyaponov_candidate

    def set_lyapunov_derivative_candidate\
        (self, lyaponov_derivative_candidate:callable):
        '''
        This method sets the lyaponov candidate derivative function for the system.
        This function is added as a countour plot in the phase portrait.

        Args:
            lyaponov_derivative_candidate (callable): A callable function that takes in the 
            current state of the system  as an array x and 
            returns the lyaponov candidate derivative function value at that point.

        Returns:
        '''

        self.lyaponov_derivative_candidate = lyaponov_derivative_candidate


    
    def test_point(self, testpoint):
        '''
        This method tests a single point in the phase portrait.

        Args:
            testpoint (np.ndarray): The point to be tested

        Returns:
            x_points (np.ndarray): The x values of the solution.
            y_points (np.ndarray): The y values of the solution.
            z_points (np.ndarray): The z values of the solution.
        '''

        if self.system_type == '2D':

            sol = solve_ivp(lambda t, x: self.system(x), [0, 100], testpoint, 
                            tspan = np.linspace(0, 100, 10000), min_step = 1e-3)

            x_points = sol.y[0,:]
            y_points = sol.y[1,:]

            neg_sys = lambda x: -1*self.system(x)
            sol_neg = solve_ivp(lambda t, x: neg_sys(x), [0, 100], 
                                testpoint, tspan = np.linspace(0, 100, 10000),
                                min_step = 1e-3)    

            x_points_neg = sol_neg.y[0,:]
            y_points_neg = sol_neg.y[1,:]

            x_points_neg = np.flip(x_points_neg)
            y_points_neg = np.flip(y_points_neg)

            x_points = np.append(x_points_neg, x_points)
            y_points = np.append(y_points_neg, y_points)

            return x_points, y_points, np.zeros(x_points.shape)
        
        elif self.system_type == '3D':

            sol = solve_ivp(lambda t, x: self.system(x), [0, 2], testpoint, 
                            tspan = np.linspace(0, 5, 10000))

            x_points = sol.y[0,:]
            y_points = sol.y[1,:]
            z_points = sol.y[2,:]

            neg_sys = lambda x: -1*self.system(x)
            sol_neg = solve_ivp(lambda t, x: neg_sys(x), [0, 10], 
                                testpoint, tspan = np.linspace(0, 10, 10000))

            x_points_neg = sol_neg.y[0,:]
            y_points_neg = sol_neg.y[1,:]
            z_points_neg = sol_neg.y[2,:]

            x_points_neg = np.flip(x_points_neg)
            y_points_neg = np.flip(y_points_neg)
            z_points_neg = np.flip(z_points_neg)

            x_points = np.append(x_points_neg, x_points)
            y_points = np.append(y_points_neg, y_points)
            z_points = np.append(z_points_neg, z_points)

            return x_points, y_points, z_points



    def generate_phase_portrait(self, x_range:tuple = (-1,1), 
                                y_range:tuple = (-1, 1), 
                                z_range:tuple = (-1, 1),
                                num_points:int = 20,
                                Title:str = 'Phase Portrait',
                                xlabel:str = 'X',
                                ylabel:str = 'Y',
                                zlabel:str = 'Z',
                                lyaponov_contour:bool = True,
                                lyaponov_derivative_contour:bool = False):
        '''
        This method generates the phase portrait of the system of ODEs.

        Args:
            x_range (tuple): The range of x values to be plotted.
            y_range (tuple): The range of y values to be plotted.
            z_range (tuple): The range of z values to be plotted.
            num_points (int): the resolution of the quiver plot. 
                              The plot will have num_points^2/3 points.
            Title (str): The title of the phase portrait.
            xlabel (str): The x-axis label.
            ylabel (str): The y-axis label.
            zlabel (str): The z-axis label.

        Returns:
            ax (matplotlib.axes): The axes object of the plot.
        '''

        ax = plt.figure()
        ax = plt.subplot(111)

        if self.system_type == '2D':

            x = np.linspace(x_range[0], x_range[1], num_points)
            y = np.linspace(y_range[0], y_range[1], num_points)

            X, Y = np.meshgrid(x, y)

            U = np.zeros(X.shape)
            V = np.zeros(Y.shape)

            Vxx = np.zeros_like(X) # lyaponov candidate function evaluation
            Vyy = np.zeros_like(Y) # lyaponov candidate function evaluation

            # lyaponov candidate derivative function evaluation
            dVxx = np.zeros_like(X) 
            dVyy = np.zeros_like(Y)

            for i in range(0, len(X)):
                
                for j in range(0, X.shape[1]):

                    gradient = self.system([X[i,j], Y[i,j]])
                    U[i,j] = gradient[0]
                    V[i,j] = gradient[1]

                    if self.lyaponov_candidate is not None:

                        Vxx[i,j] = self.lyaponov_candidate([X[i,j], Y[i,j]])
                        Vyy[i,j] = self.lyaponov_candidate([X[i,j], Y[i,j]])

                    if self.lyaponov_derivative_candidate is not None:

                        dVxx[i,j] = self.lyaponov_derivative_candidate([X[i,j], 
                                                                        Y[i,j]])
                        
                        dVyy[i,j] = self.lyaponov_derivative_candidate([X[i,j], 
                                                                        Y[i,j]])

            ax.quiver(X, Y, U, V, alpha = 0.25)
            plt.xlim(x_range)
            plt.ylim(y_range)

            for i in range(0, len(self.test_points)):
                
                x_points, y_points,_ = self.test_point(self.test_points[i])
                ax.plot(x_points, y_points, 'r', alpha = 0.33)

                # Make an arrow at the IC of the trajectory:

                index = int(len(x_points)/2) - 2
                xlen = x_points[index] - x_points[index - 1]
                ylen = y_points[index] - y_points[index - 1]

                xlen = xlen / np.sqrt(xlen**2 + ylen**2) * 0.05
                ylen = ylen / np.sqrt(xlen**2 + ylen**2) * 0.05

                ax.arrow(x_points[index], y_points[index], 
                         xlen, ylen, 
                         head_width = 0.05, 
                         head_length = 0.05,
                         width = 0.,
                         fc = 'r', ec = 'r',
                         alpha = 0.1666,
                         head_starts_at_zero = True,
                         length_includes_head = True)
            
            if lyaponov_contour and self.lyaponov_candidate is not None:

                CS = ax.contour(X, Y, Vxx, levels = 10, alpha = 0.75)
                ax.clabel(CS, inline = True, fontsize = 10)

            if lyaponov_derivative_contour and \
               self.lyaponov_derivative_candidate is not None:

                CS = ax.contour(X, Y, dVxx, levels = 10, alpha = 0.75)
                ax.clabel(CS, inline = True, fontsize = 10)

            for i in range(0, len(self.eq_points)):

                ax.plot(self.eq_points[i][0], self.eq_points[i][1], 'rx')

            ax.set_title(Title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        elif self.system_type == '3D':

            x = np.linspace(x_range[0], x_range[1], num_points)
            y = np.linspace(y_range[0], y_range[1], num_points)
            z = np.linspace(z_range[0], z_range[1], num_points)

            X, Y, Z = np.meshgrid(x, y, z)

            U = np.zeros(X.shape)
            V = np.zeros(Y.shape)
            W = np.zeros(Z.shape)

            for i in range(0, len(X)):
                
                for j in range(0, X.shape[1]):

                    for k in range(0, X.shape[2]):

                        gradient = self.system([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                        U[i,j,k] = gradient[0]
                        V[i,j,k] = gradient[1]
                        W[i,j,k] = gradient[2]
            
            ax = plt.axes(projection = '3d')
            ax.quiver(X, Y, Z, U, V, W, length = 0.1, normalize = True)

            for i in range(0, len(self.test_points)):
                x_points, y_points, z_points = self.test_point(self.test_points[i])
                ax.plot(x_points, y_points, z_points, 'r')


            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_zlim(z_range)
            ax.set_title(Title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)


        
        return ax