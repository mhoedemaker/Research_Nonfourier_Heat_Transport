import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
import os
from tabulate import tabulate
import scipy.stats as stats
import pdb
import random

# simulation parameters
n = 1372
steps = 800
delta = 0.005
m = 1
mass = 1 # 6.6335 * 10**(-26) # kg Argon
e = 1#1.66*10**(-21)
sigma = 1#3.4 * 10**(-10) # A # 3.4 # Argon
kb = 1#1.380649*10**(-23) #J/K
n = 10 # number of particles
cutoff = 2.5#3*sigma #m
T1 = 1.380649*10**(-23)*125/(1.66*10**(-21)) #00 # K Temperature to set the system
T2 = 1.380649*10**(-23)*300/(1.66*10**(-21))
K1 = 3/2*T1*(n-1)*kb
K2 = 3/2*T2*(n-1)*kb
T_damp = 0.05 # damping coefficient
L = 6.8#2.57*10**(-9) #(m*n/950)**(1/3)#6.8 # length of bounding box
V = L**3 # volume of the system

############################################# The Code Extracts all the Data ####################################################
#Reads initial non-dimensionalized positions from input file
fpath = "C:/Users/madel/Documents/Mcgaughey"
data = np.loadtxt('C:/Users/madel/Documents/Mcgaughey/particle_def/1D_chain_long.txt')
#for each of the files

x = data[:,0]
y = data[:,1]
z = data[:,2]

#Initializes particle velocities to zero
v_x = np.zeros_like(x)
v_y = np.zeros_like(y)
v_z = np.zeros_like(z)
depth = 0

def init_positions(x,y,z):
    particles = []
    for i in range(len(x)):
        particles += [[x[i],y[i],z[i]]]
    return particles

def recursiveDivide(n,newL):
    length = len(newL)
    if length >= n:
        if length > n:
            print("help this is an error")
        return np.array(newL)
    else:
        rd = np.random.randint(length)
        val = newL[rd]
        n1 = val * np.random.random()
        n2 = val - n1
        newerL = newL
        newerL.pop(rd)
        newerL.append(n1)
        newerL.append(n2)
        return recursiveDivide(n,newerL)

def init_vel():
    vel = np.zeros((n, 3))
    sumV2_1 = 2*K1/m
    sumV2_2 = 2*K2/m # vx^2 + vy^2 + vz^2
    mean_x = np.mean(vel[:,0])
    mean_y = np.mean(vel[:,1])
    mean_z = np.mean(vel[:,2])
    # nondimensionalizing initial velocity
    newVel1 = recursiveDivide(3,[sumV2_1])
    newVel2 = recursiveDivide(3,[sumV2_2])
    for atom in range(0, n):
        if atom == 0 or atom == 1 or atom == 2:
            vel[atom,0] = newVel1[atom]
        if atom == n-1 or atom == n-2 or atom == n-3:
            vel[atom,0] = -newVel2[n-1 - atom]
        vel[atom, 0] -= mean_x
        vel[atom, 1] -= mean_y
        vel[atom, 2] -= mean_z
    return vel

# perturb all the particles by a random number between 0 and 1
# Accept the perturbation at different frequencies (e.g. if the temperature is higher, use a different gaussian curve) for ends versus the middle
# p = e^(perturbed-current)/kbT
# Assign each molecule a temperature (such that the ones in the temperature bath stay the same) and the ones in the middle can change temperature
# We are still going to lose the molecule on the ends
# Have to go more in-depth to hadjiconstantinou work
# why monte carlo? if I do a fixed temperature of the middle part of the chain and the hot part of the chain begins causing acceleration, can I transmit that as temperature somehow?
# Start them at a lower temperature, then use the change in energy from colliding with the faster atom to temporarily change the temperature (if it accepts the move at higher probability, the neighboring atom has to be able to accept that
#too because momentum is conserved)
# There has to be some term for resistance or the heat will not flow correctly.... it can't just be perfect momentum conservation and there has to be a heat sink and source
# MD simulations do not perform perfect collisions...so resistance should be there.

# perturbs the 
def monteCarlo():

    return







######################### Monte Carlo #################################

beta = 5
epochs = 5000

perturb = random.random()*random.choice([-1,1])

def plotConvergence(x,title=""):
    expected = [0]
    for i in range(1,len(x)):
        expected += [np.sum(x[0:i])/i]
    plt.figure()
    plt.title(title)
    t = np.linspace(0,epochs,len(x))
    plt.scatter(t,x,label = "variable") # time window is approximated (and irrelevant)
    plt.plot(t,expected,label = "expected value",color = "red")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()















