import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import os
from tabulate import tabulate
import scipy.stats as stats
import pdb
import random
################################ Question 5 ##################################

################ Function #################
#   This file initiates all the particles
#   It performs the iterations for each time step and calls the position, velocity, and force and functions on the particle
#   It generates a txt file in .xyz format for each iteration

#Reads initial non-dimensionalized positions from input file
fpath = "C:/Users/madel/Documents/Mcgaughey"
data = np.loadtxt('C:/Users/madel/Documents/Mcgaughey/particle_def/1D_chain_long.txt')

#Defines parameters for velocity Verlet, time increment, number of steps
# If you want periodic boundary conditions, use the cutoff > L/2
delta = .001
steps =  1000
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
continuousFCutoff = True # if you want to implement a continuous force cutoff
boundingBox =True # if you want to create a bounding box. Automatically implements nearst-image convention
noseHoover = False
plotDisplacement = False
offset = 0.3 # whatever works so the kinetic energy equalizes at T = 100
window_size = 0.5/delta
Equilibrium = 0 # time the system reaches equilibrium (if we want to set this to vary density)
###################################### This code sets up the simulation #####################################################

#for each of the files
paramLst = []

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

# Note: K1 and K2 are too large if they are really -10 and 5

####################################### The following code is used to perform one iteration of Verlet ##############################################
# calculates the distance between two particles, p1 and p2
def distance(p1,p2):
    rx = p2[0] - p1[0]
    ry = p2[1] - p1[1]
    rz = p2[2] - p1[2]
    r = np.sqrt((rx)**2+(ry)**2+(rz)**2)
    return r,rx,ry,rz


def calculateAllForcesAndPotentials(positions):
    atom_count = len(positions)
    forces = np.zeros((atom_count, 3))
    sumrF = 0
    potentials = np.zeros(atom_count)
    t_c = (sigma/cutoff)**6
    F_c = e*(48*t_c**2 - 24*t_c)/cutoff
    # for each interaction
    for atom in range(atom_count):
        for other in range(atom + 1, atom_count):
            p1 = positions[atom]
            p2 = positions[other]
            if atom != other:
                r, rx, ry, rz = distance(p1, p2)
                # rewrite force for continuous cutoff
                if continuousFCutoff == True:
                    if r < (cutoff):
                        F = e * ((48* (sigma/r)**13) - (24* (sigma/r)**7)) - e * (((48*(sigma/cutoff)**13) - (24* (sigma/cutoff)**7)))
                        sumrF = sumrF + F*r
                        U = e * ((4 *(sigma/r)**12) - (4*(sigma/r)**6)) - e * ((4*(sigma/cutoff)**12 - (4*(sigma/cutoff)**6))) + e * (r-cutoff)*(48*(sigma/cutoff)**13 - 24*(sigma/cutoff)**7)
                    else:
                        F = 0
                        U = 0
                else:
                    F = e * ((48*(sigma/r)**13) - (24 * (sigma/r)**7)) 
                    U = e *((4 * (sigma/r)**12) - (4 * (sigma/r)**6))
                    sumrF += r*F
                potentials[atom] += U   
                Fx = -1*F * (rx / r)
                Fy = -1*F * (ry / r)
                Fz = -1*F * (rz / r)            
                forces[atom, 0] += Fx
                forces[other, 0] -= Fx
                forces[atom, 1] += Fy
                forces[other, 1] -= Fy
                forces[atom, 2] += Fz
                forces[other, 2] -= Fz 
    return forces, sumrF, potentials

# updates the velocities based on force (for verlet calculations)
def update_vel(v, f):
    v_new = np.zeros(3)
    for dir in range(0, 3):
        v_new[dir] = v[dir] + (delta / 2) * (f[dir] / m)
    return v_new

# updates the positions based on velociy (for the verlet calculations)
def update_pos(p, v):
    p_new = np.zeros(3)
    for dir in range(0, 3):
        p_new[dir] = p[dir] + delta * v[dir]
    return p_new

# calculates the potential energy of the entire system for each time step (adding potentials for each particle)
# calculates the kinetic energy for the system for each time step and the momentum
def kinetic(velocities):
    e_sum = 0
    px_sum = 0
    py_sum = 0
    pz_sum = 0
    for i in range(len(velocities)):
        p = velocities[i]
        e_new = (1 / 2) * m * ((p[0])**2+(p[1])**2+(p[2])**2)
        e_sum += e_new
        px_sum += m * p[0]
        py_sum += m * p[1]
        pz_sum += m * p[2]
    T_inst = 2*e_sum/(3*(n-1)*kb) 
    return e_sum, px_sum,py_sum,pz_sum, T_inst # returns an np array of total energy out and an float for momentum in x,y,z directions

# Calculates the hamiltonian for each time step
def calculateHamiltonian(e, pot):
    h = []
    for timestep in range(0, steps):
        h.append(e[timestep] + pot[timestep])
    return h

# performs velocity verlet
# note positions and velocities are stored within each particle in a list of particles while force is a matrix of fx,fy,fz for each particle
def Verlet(positions,velocities, f,langevin):
    #atom_count = len(particles)
    lamda = 0.01 # this is completely made up
    #print("velocitiesss", velocities)
    if langevin != True:
        for atom in range(0, n):
            p = positions[atom]
            v = velocities[atom]
            f_init = f[atom]
            # update velocity
            velocities[atom][0]= v[0] + (delta / 2) * (f_init[0] / mass)
            # update position
            positions[atom][0] = p[0] + delta * v[0]
                 
        f_out, rFsum, potentials = calculateAllForcesAndPotentials(positions)
        for atom in range(0, n):
            velocities[atom][0] = v[0]+ (delta / 2) * (f_out[atom,0] / mass)
    else:
        for atom in range(3,n-3):
            # update velocity
            p = positions[atom]
            v = velocities[atom]
            f_init = f[atom]
            velocities[atom][0]= v[0] + (delta / 2) * (f_init[0] / mass)
            # update position
            positions[atom][0] = p[0] + delta * v[0]
        for atom in range(0,3):
            p = positions[atom]
            v = velocities[atom]
            f_init = f[atom]
            rand_force = np.sqrt(2*lamda*m*kb*T1/delta)*random.random()
            velocities[atom][0]= v[0] + (delta / (2* mass)) * (f_init[0] - lamda*mass*v[0]+rand_force)    
            # update position
            positions[atom][0] = p[0] + delta * v[0]
        
        for atom in range(n-3,n):
            p = positions[atom]
            v = velocities[atom]
            f_init = f[atom]
            rand_force = np.sqrt(2*lamda*m*kb*T2/delta)*random.random()
            velocities[atom][0]= v[0] + (delta / (2* mass)) * (f_init[0] - lamda*mass*v[0]+rand_force)   
            # update position
            positions[atom][0] = p[0] + delta * v[0]
 
        #print("positionssss", positions)
        f_out, rFsum, potentials = calculateAllForcesAndPotentials(positions)
        for atom in range(3, n-3):
            v = velocities[atom]
            velocities[atom][0] = v[0]+ (delta / 2) * (f_out[atom,0] / mass)
        #Apply Langevin thermostat
        for atom in range(0,3):
            v = velocities[atom]
            rand_force = np.sqrt(2*lamda*m*kb*T1/delta)*random.random()
            velocities[atom][0]= v[0] + (delta / (2* mass)) * (f_init[0] - lamda*mass*v[0]+rand_force)          
        for atom in range(n-3,n):
            v = velocities[atom]
            rand_force = np.sqrt(2*lamda*m*kb*T2/delta)*random.random()
            velocities[atom][0]= v[0] + (delta / (2* mass)) * (f_init[0] - lamda*mass*v[0]+rand_force)         
    return f_out,rFsum

########################################  This code runs the simulation - controls everything with time #####################################################

# Moves the particles in whatever direction they are forced
def performVerlet(positions,velocities,steps,langevin=True):
    all_forces = 0
    inst_Temp = []
    inst_Press = []
    all_kinetic = []
    momentum = []
    all_potentials = []
    F,rFsum,potentials= calculateAllForcesAndPotentials(positions)
    U = np.sum(potentials)
    avgU = U
    avgP = 0
    K,_,_,_,T_inst = kinetic(velocities)
    #for each step in time
    t = 0
    avgU_lst = []
    # This part calculates the total potential and kinetic energy for each step
    for i in range(1,steps):
        if i % 100 == 0:
            print('step', i)
        #Perform velocity verlet for each step  
        F,rFsum = Verlet(positions,velocities,F,langevin)
        all_forces += F
        #all_potentials += [U]
        #all_kinetic += [K]
        #if i > 50:
           # avgU = (avgU*(i+1)+U)/(i+2)
           # avgU_lst += [avgU]

        generatetxt(i,positions)
        for i in range(len(positions)):
            p = positions[i]
    return all_forces, all_potentials, all_kinetic, momentum, inst_Press, inst_Temp, avgU_lst


#################################################### Determine Equibrilation ########################################################
def computeVariance(data):
    mean_sig = np.mean(data)
    var = 0
    for i in range(len(data)):
        var += (data[i] - mean_sig)**2
    return var/len(data)

def isEquilibrated(inst_T,inst_P):
    for i in range(int(len(inst_T)-window_size)):
        data = inst_T[i:int((i+window_size))]
        var = computeVariance(data)
        if var < 0.1:
            print("equilibrated")
            return i*delta
    return False

############################################ Creates graphical depictions of results ##################################################

def generatetxt(i,positions):
    fname = 'C:/Users/madel/Documents/Mcgaughey/particle_def/10_test.txt' 
    f = open(fname,"a")
    f.write("%i\n\n"%n)
    for p in positions:
        f.write("H")
        f.write("\t%4.3f"%p[0])
        f.write("\t%4.3f"%p[1])
        f.write("\t%4.3f\n"%p[2])
    f.close()
    return 

def plotEnergies(particles,Energies):
    p = particles[0]
    times = p.ttrace
    plt.figure()
    plt.plot(times,Energies[0],label="potential")
    plt.plot(times,Energies[1],label="kinetic")
    plt.plot(times,Energies[2],label="hamiltonian")
    plt.legend()
    plt.title("Energy over time")
    plt.show()

# plot of momentum over time given particle velocities
def plotMomentum(momentums,particles):
    p = particles[0]
    times = p.ttrace
    px = np.array(momentums)[:,0]
    py = np.array(momentums)[:,1]
    pz = np.array(momentums)[:,2]

    plt.figure()
    plt.plot(times,px,label = "px")
    plt.plot(times, py,label = "py")
    plt.plot(times,pz,label = "pz")
    plt.legend()
    plt.title("Momentum over Time")
    plt.show()

def plotTemperature(inst_Temp,particles):
    p = particles[0]
    times = p.ttrace
    plt.figure()
    plt.plot(times,inst_Temp,label = "temperature")
    plt.title("Instantaneous Temperature")
    plt.show()

def plotPressure(inst_Press,particles):
    p = particles[0]
    times = p.ttrace
    plt.figure()
    plt.plot(times,inst_Press,label = "pressure")
    plt.title("Instantaneous Pressure")
    plt.show()

def tabulateQuant(potential,kineticE,inst_press,inst_Temp):
    header = ["U","K","T_inst","P_inst"]
    if steps > 100:
        avgT = np.average(inst_Temp[(len(inst_Temp)-100):len(inst_Temp)])
        avgP = np.average(inst_press[(len(inst_press)-100):len(inst_press)])
        avgU = np.average(potential[(len(potential)-100):len(potential)])
        avgK = np.average(kineticE[(len(kineticE)-100):len(kineticE)])
        table = [["U",avgU],["K",avgK],["T",avgT],["P",avgP]]
        print(tabulate(table))
    return

def plotParticleX(particles,bc):
    x0 = []
    x10 = []
    x100 = []
    for particle in particles:
        xlist = particles[particle].xtrace
        x0 += xlist[0]
        x10 += xlist[10]
        x100 += xlist[100]
    fig, ax = plt.subplots(nrows=2, ncols=2)

    

################################################### Initialize Code ################################################################
positions = init_positions(x,y,z)
F,rFsum,potentials= calculateAllForcesAndPotentials(positions)
velocities = init_vel()
all_forces,all_potentials,all_kinetic,momentum,inst_Press,inst_Temp,avgU_lst = performVerlet(positions,velocities,steps,langevin=True)

generatetxt([1,2,3,4,5,6,7,8,9,10],positions)