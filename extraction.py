import numpy as np
import matplotlib.pyplot as plt
from particle_2 import particle
import tkinter as tk
import os
from tabulate import tabulate
import scipy.stats as stats
from scipy.optimize import curve_fit
import pdb
import random
################################ Question 5 ###################################

# When you nondimensionalize, what do you need to nondimensionalize?
# What am I doing with temperature on the ends of the chain? Do I leave only one molecule oscillating? 

################ Function #################
#   This file initiates all the particles
#   It performs the iterations for each time step and calls the position, velocity, and force and functions on the particle
#   It generates a txt file in .xyz format for each iteration

#Reads initial non-dimensionalized positions from input file
fpath = "C:/Users/madel/Documents/Mcgaughey"
data = np.loadtxt('C:/Users/madel/Documents/Mcgaughey/particle_def/1D_chain.txt')

#for each of the files
paramLst = []

x = data[:,0]
y = data[:,1]
z = data[:,2]
print(z)
