# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp

#system of ODEs, that once solved give the coordinates for each individual hair
def dx_ds(theta):
    return np.cos(theta) * np.cos(phi)
def dy_ds(theta):
    return -np.cos(theta) * np.sin(phi)
def dz_ds(theta):
    return np.sin(theta)

#BVP; system of ODEs giving angles for each individual hair
def d2theta_ds2(s,theta,dtheta,fg,fx):
    return s*fg*np.cos(theta) + s*fx*np.sin(theta)*np.cos(phi)
def d2phi_ds2(s,theta,dtheta,fg,fx):
    return -s * fx * np.sin(phi)*np.sin(theta)

def shooting_method(d2theta_ds, interval, bcs, N, fg, fx):
    """
    Solve BVP using shooting method. Converts a BVP to an IVP by guessing 
    additional intial condition. Solve the IVP and check agreement with end point.
    Modify intial guess and repeat until solved (use root finding method to find best guess).
    
    Here the Brentq method is used to find root, which is a hybrid method. 
    Combines the bisection method, the secant method and inverse quadratic interpolation.
    Uses the secant method or inverse quadratic interpolation to quickly find root,
    which are fast-converging but less reliable but falls back on the more 
    robust bisection method if necessary. It has the reliability of bisection 
    but it can be as quick as some of the less-reliable methods. 
    
    Shooting method is quick and accurate (when it works). However it comes
    with some challenges. For instance, it fails to excatly satisfy the bcs at the end
    of the interval and there and there are problems for which the BVP is well-posed 
    but the corresponding IVP for the shooting method is not (e.g. IVP has infinite solutions).
   (https://nbviewer.org/github/IanHawke/NumericalMethods/blob/master/Lectures/18%20-%20Shooting.ipynb)

    Parameters
    ----------
    d2theta_ds : Function, BVP function.
    interval : List of 2 floats, start and end points (0,L).
    bcs : List of 2 floats, boundary condition values
    N : Integer, grid size.
    fg : Float, force due to gravity.
    fx : Float, force due to wind.

    Returns
    -------
    Array of floats, solution at grid points.

    """
    assert N>1, "Number must be greater than 1"
    assert (int(N) == N), "Number of steps must be an integer"
    assert len(interval) ==2,"Interval must be a 2 element array"
    
    def rhs(s, z):
        """
        Rewrite equation as a first-order system and implement its right-hand side evaluation

        Parameters
        ----------
        s : float, location along hair.
        z : list of functions, [y, y']

        Returns
        -------
        Array of functions, right hand side implementation

        """
        theta, dtheta_ds = z
        return np.array([dtheta_ds, d2theta_ds(s, theta, dtheta_ds, fg, fx)])

    def residual(guess):
        """
        Implement evaluation of the boundary condition residuals.
     
        Parameters
        ----------
       
        guess: Float, guess for derivative of solution.
     
        Returns
        -------
       
        Float, residual of shooting solution.
        """
        #Convert BVP to IVP by guessing additional initial conditions, bcs[0]
        shoot = solve_ivp(rhs, interval, [bcs[0], guess])
        #check agreement with conditions for end point, bcs[1]
        return shoot.y[1, -1] - bcs[1]
    
    #Find root using brentq method.
    root = scipy.optimize.brentq(residual, -5, 5) #find where fβ−fb=0
    #grid mesh, with N nodes
    s = np.linspace(interval[0], interval[-1], N)
    #run the solver
    sol = solve_ivp(rhs, interval, [bcs[0], root], t_eval=s)
    return sol.y

#%% Task 1

def hair_coordinates(L, R, fx, fg, theta0):
    """
    Finds the x and z coordinates of the hair strands for a 2D case. Solves a BVP
    to determine θ(s) for each strand of hair and then the coordinates of the 
    hair are found by solving a system of ODEs.

    Parameters
    ----------
    L : Float, hair length.
    R : Float, radius of sphere/head.
    fx : Float, parameter describing the force from gravity relative to the bending stiffness.
    fg : Float, parameter describing the force from the wind relative to the bending stiffness.
    theta0 : List of floats, specify where the hairs meet the head.

    Returns
    -------
    Array of floats, returns the (x, z) coordinates of the hairs.
    """
    assert ((not np.isnan(L))),"L must be non-zero."
    assert ((not np.isnan(R))),"R must be non-zero."
    
    hair_x_z = [] #allocate memory for hair x and z coordinates
    N = 10 #number of s; grid points
    #for each theta0:
    for theta in theta0:
        #shooting solution
        sol = shooting_method(d2theta_ds2, [0,L], [theta,0], N, fg, fx)
        #create s grid (0 to L)
        s = np.linspace(0, L, N) 
        
        #The spatial positions of the hair that joins the head (at s = 0) of each theta0
        x0 = R*np.cos(theta)
        z0 = R*np.sin(theta)

        def reduced_order(s, z):
            #convert into a first order system, y''= f(x,y,y')
            x,z,theta,dtheta_ds = z
            return np.array([dx_ds(theta), dz_ds(theta), dtheta_ds, d2theta_ds2(s, theta, dtheta_ds, fg, fx)])
        #Solve for the x and z coordinates
        hairLoc = solve_ivp(reduced_order, [0,L],[x0, z0, sol[0][0], sol[1][0]], t_eval=s)
        x, z = hairLoc.y[0], hairLoc.y[1]
        #Appending x and z coordinates to array
        hair_x_z.append([x, z])
    return hair_x_z

#%% Task 2 - Compute and plot the location of 20 different hairs, fx=0

f_g = 0.1 #cm^−3, force from gravity relative to the bending stiffness
f_x = 0 #cm^−3, force from the wind relative to the bending stiffness
R = 10 #cm, radius of sphere/head
L = 4  #cm, hair length
phi =0 #radians, longitude of hair(φ = 0)
theta_0 = np.linspace(0, np.pi, num=20) #radians, latitude of hair(θ)

#find hair coordinates
hair_coord = hair_coordinates(L, R, f_x, f_g, theta_0)

#Plot hair
#for each hair strand
for hair in hair_coord:
    #hair[0]==> x coordinates
    #hair[1]==> z coordinates
    plt.plot(hair[0], hair[1], "k")

#plot circle
theta = np.linspace(0, 2*np.pi, 100)
x1 = R*np.cos(theta)
x2 = R*np.sin(theta)
plt.plot(x1,x2)

#formating and labelling
plt.title("No Wind, fx = 0, fg = 0.1")
plt.xlabel("x"); plt.ylabel("z")
plt.show()

#%% Task 3 - Compute and plot the location of 20 different hairs, fx = 0.1

f_g = 0.1 #cm^−3, force from gravity relative to the bending stiffness
f_x = 0.1 #cm^−3, force from the wind relative to the bending stiffness
R = 10 #cm, radius of sphere/head
L = 4  #cm, hair length
phi =0 #radians, longitude of hair(φ = 0)
theta_0 = np.linspace(0, np.pi, num=20) #radians, latitude of hair(θ)

#find hair coordinates
hair_coord = hair_coordinates(L, R, f_x, f_g, theta_0)

#Plot hair
#for each hair strand
for hair in hair_coord:
    #hair[0]==> x coordinates
    #hair[1]==> z coordinates
    plt.plot(hair[0], hair[1], "k")

#plot a circle
theta = np.linspace(0, 2*np.pi, 100)
x1 = R*np.cos(theta)
x2 = R*np.sin(theta)
plt.plot(x1,x2)

#formating and labelling
plt.title("Wind, fx = 0.1, fg = 0.1")
plt.xlabel("x"); plt.ylabel("z")
plt.show()


