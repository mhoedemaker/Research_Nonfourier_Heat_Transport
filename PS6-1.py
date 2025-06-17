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

################ Function #################
#   This file initiates all the particles
#   It performs the iterations for each time step and calls the position, velocity, and force and \\\\\\\\\\\\\\\\\\\\ functions on the particle
#   It generates a txt file in .xyz format for each iteration

#Reads initial non-dimensionalized positions from input file
fpath = "C:/Users/madel/Documents/Mcgaughey/particle_def"
data = np.loadtxt('C:/Users/madel/Documents/Mcgaughey/particle_def/liquid256.txt')

#for each of the files
paramLst = []

x = data[:,0]
y = data[:,1]
z = data[:,2]

#Defines parameters for velocity Verlet, time increment, number of steps
# If you want periodic boundary conditions, use the cutoff > L/2
delta = .001
steps =  1000
m = 1
mass = 6.6335 * 10**(-26) # kg Argon
e = 1#1.66*10**(-21)
sigma = 1#3.4 * 10**(-10) # A # 3.4 # Argon
kb = 1#1.380649*10**(-23) #J/K
cutoff = 2.5#3*sigma #m
T = 1.380649*10**(-23)*125/(1.66*10**(-21)) #00 # K Temperature to set the system
print("T", T)
T_damp = 0.05 # damping coefficient
n = 256 # number of particles
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

#Initializes particle velocities to zero
v_x = np.zeros_like(x)
v_y = np.zeros_like(y)
v_z = np.zeros_like(z)
depth = 0
# divides a list [sum] into n pieces such that the sum of all values in the list is sum
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

# initiates particle velocities such that momentum is conserved and kinetic energy is set based on T
def randMomentum(n,T):
    #T1= kb*T/e
    K = (T)/2*3*(n-1)*kb  # kinetic energy constraint
    print("K", K)
    sumV2 = 2*K/m # vx^2 + vy^2 + vz^2
    halfVel = recursiveDivide(np.floor(n/2),[sumV2/2]) #initialized the velocities of half the particles to V2/2
    maxV = np.max(halfVel)

    vi_x = []
    vi_y = []
    vi_z = []

    # for each velocity, create a particle x,y,z velocity
    for v in halfVel:
        [vx2,vy2,vz2] = recursiveDivide(3,[v])
        vi_x += [np.sqrt(vx2)]
        vi_y += [np.sqrt(vy2)]
        vi_z += [np.sqrt(vz2)]
        vi_x += [-1*np.sqrt(vx2)]
        vi_y += [-1*np.sqrt(vy2)]
        vi_z += [-1*np.sqrt(vz2)]

    if n%2 != 0:
        vi_x += [0]
        vi_y += [0]
        vi_z += [0]

    return vi_x,vi_y,vi_z,maxV

def init_vel():
    vel = np.zeros((n, 3))
    for atom in range(0, n):
        vel[atom] = np.random.normal(0, np.sqrt(T), 3)
    mean_x = np.mean(vel[:,0])
    mean_y = np.mean(vel[:,1])
    mean_z = np.mean(vel[:,2])
    for atom in range(0, n):
        vel[atom, 0] -= mean_x
        vel[atom, 1] -= mean_y
        vel[atom, 2] -= mean_z
    return vel[:,0],vel[:,1],vel[:,2],np.max(vel)


#Create all particles defined by position, velocity, and time
def createParticles(x,y,z,v_x,v_y,v_z):
    particles = []
    for i in range(n):
        x0 = x[i]
        y0 = y[i]
        z0 = z[i]
        v_x0 = v_x[i]
        v_y0 = v_y[i]
        v_z0 = v_z[i]
        number = int(i)
        particles += [particle(number,x0,y0,z0,v_x0,v_y0,v_z0)]
    return particles

# calculates the coordinates of the initial center of m of the system
def calcCOM(x,y,z):
    x_m = 0
    y_m = 0
    z_m = 0
    for i in range(n):
        x_m += x[i] * m
        y_m += y[i] * m
        z_m += z[i] * m
    x_m = x_m / (n*m)
    y_m = y_m / (n*m)
    z_m = z_m / (n*m)
    return [x_m,y_m,z_m]

# calculates the distance to the nearest boundary
def calcDistanceToL(p,L):
    dx = p.x-L
    dy = p.y - L
    dz = p.z - L
    return [dx,dy,dz]

def createBoundaryCond(COM,L):
    [x_m,y_m,z_m] = COM
    left = x_m - L/2
    right = x_m + L/2
    bottom = y_m - L/2
    top = y_m + L/2
    up = z_m + L/2
    down = z_m - L/2
    return [left,right,top,bottom,up,down]

####################################### The following code is used to perform one iteration of Verlet ##############################################

# calculates the distance between two particles, p1 and p2
def distance(p1,p2):
    rx = p2.x - p1.x
    ry = p2.y - p1.y
    rz = p2.z - p1.z
    r = np.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2)
    return r,rx,ry,rz

def calculateAllForcesAndPotentials(particles):
    atom_count = len(particles)
    forces = np.zeros((atom_count, 3))
    sumrF = 0
    atom_count = len(particles)
    potentials = np.zeros(atom_count)
    t_c = (sigma/cutoff)**6
    F_c = e*(48*t_c**2 - 24*t_c)/cutoff
    # for each interaction
    for atom in range(atom_count):
        p1 = particles[atom]
        for other in range(atom + 1, atom_count):
            p1 = particles[atom]
            p2 = particles[other]
            r, rx, ry, rz = distance(p1, p2)
            # switch direction and magnitude of force for bounding box based on nearest image convention
            if boundingBox == True:
                if np.abs(rx) > L/2:
                    if p1.x > p2.x:
                        rx = L - np.abs(rx) # (p2 + L) - p1)
                    else:
                        rx = rx - L # (p2 - L) - p1
                if np.abs(ry) > L/2:
                    if p1.y > p2.y:
                        ry = L - np.abs(ry)
                    else:
                        ry = ry - L
                if np.abs(rz) > L/2:
                    if p1.z > p2.z:
                        rz = L - np.abs(rz)
                    else:
                        rz = rz - L
                r = np.sqrt(rx**2+ry**2+rz**2) 
            # rewrite force for continuous cutoff
            if continuousFCutoff == True:
                if r < (cutoff):
                    t = (sigma/r)**6
                    prefac = e *(48*t**2 - 24*t)/r - F_c
                    
                    sumrF = sumrF + prefac*r
                    F = e * ((48* (sigma/r)**13) - (24* (sigma/r)**7)) - e * (((48*(sigma/cutoff)**13) - (24* (sigma/cutoff)**7)))
                    U = e * ((4 *(sigma/r)**12) - (4*(sigma/r)**6)) - e * ((4*(sigma/cutoff)**12 - (4*(sigma/cutoff)**6))) + e * (r-cutoff)*(48*(sigma/cutoff)**13 - 24*(sigma/cutoff)**7)
                    
                    #sumrF += r*F
                    # F_c = self.e*(48*t_c**2 - 24*t_c)/self.r_cut
                    # prefac = self.e *(48*t**2 - 24*t)/rij - F_c
                else:
                    F = 0
                    U = 0
            else:
                F = e * ((48*(sigma/r)**13) - (24 * (sigma/r)**7)) 
                U = e *((4 * (sigma/r)**12) - (4 * (sigma/r)**6))
                sumrF += r*F
            potentials[atom] += U    
            #print(F)
            Fx = -1*F * (rx / r)
            Fy = -1*F * (ry / r)
            Fz = -1*F * (rz / r)            
            forces[atom, 0] += Fx
            forces[other, 0] -= Fx
            forces[atom, 1] += Fy
            forces[other, 1] -= Fy
            forces[atom, 2] += Fz
            forces[other, 2] -= Fz
        # print("sumrF",sumrF)
    
    return forces, sumrF, potentials

# calculates the forces between all particles and stores in a Nx3 array [[Fx,Fy,Fz]]
# also calculates the rF for all interactions where both particles are in the bounding box
def calculateForcesAndPotentials(particles,p):
    atom_count = len(particles)
    forces = np.zeros((atom_count, 3))
    sumrF = 0
    pot = 0
    # for each interaction
    for other in range(atom_count):
        p1 = p
        p2 = particles[other]
        if p1.id != p2.id:
            r, rx, ry, rz = distance(p1, p2)
            # switch direction and magnitude of force for bounding box based on nearest image convention
            if boundingBox == True:
                if np.abs(rx) > L/2:
                    if p1.x > p2.x:
                        rx = L - np.abs(rx) # (p2 + L) - p1
                    else:
                        rx = rx - L # (p2 - L) - p1
                if np.abs(ry) > L/2:
                    if p1.y > p2.y:
                        ry = L - np.abs(ry)
                    else:
                        ry = ry - L
                if np.abs(rz) > L/2:
                    if p1.z > p2.z:
                        rz = L - np.abs(rz)
                    else:
                        rz = rz - L
                r = np.sqrt(rx**2+ry**2+rz**2) 
            # rewrite force for continuous cutoff
            if continuousFCutoff == True:
                if r < (cutoff):
                    F = e * ((48* (sigma/r)**13) - (24* (sigma/r)**7)) - e * (((48*(sigma/cutoff)**13) - (24* (sigma/cutoff)**7)))
                    U = e * ((4 *(sigma/r)**12) - (4*(sigma/r)**6)) - e * ((4*(sigma/cutoff)**12 - (4*(sigma/cutoff)**6))) + e * (r-cutoff)*(48*(sigma/cutoff)**13 - 24*(sigma/cutoff)**7)
                else:
                    F = 0
                    U = 0
            else:
                F = e * ((48*(sigma/r)**13) - (24 * (sigma/r)**7)) 
                U = e *((4 * (sigma/r)**12) - (4 * (sigma/r)**6))
            pot += U    
            #print(F)
            # Fx = -1*F * (rx / r)
            # Fy = -1*F * (ry / r)
            # Fz = -1*F * (rz / r)

            # print(forces)
            # forces[p1.id, 0] += Fx
            # forces[other, 0] -= Fx
            # forces[p1.id, 1] += Fy
            # forces[other, 1] -= Fy
            # forces[p1.id, 2] += Fz
            # forces[other, 2] -= Fz
    U = pot
    return sumrF, U

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
def kinetic(particles):
    e_sum = 0
    px_sum = 0
    py_sum = 0
    pz_sum = 0
    prevK = 0
    prevV = []
    for i in range(len(particles)):
        p = particles[i]
        e_new = (1 / 2) * m * ((p.vx)**2+(p.vy)**2+(p.vz)**2)
        e_sum += e_new
        px_sum += m * p.vx
        py_sum += m * p.vy
        pz_sum += m * p.vz
    T_inst = 2*e_sum/(3*(n-1)*kb) 
    return e_sum, px_sum,py_sum,pz_sum, T_inst # returns an np array of total energy out and an float for momentum in x,y,z directions

# Calculates the hamiltonian for each time step
def calculateHamiltonian(e, pot):
    h = []
    for timestep in range(0, steps):
        h.append(e[timestep] + pot[timestep])
    return h

# generates a random perturbation
def perturb():
    x = np.random.random()*2* np.random.choice([-1,1])
    y = np.random.random()*2* np.random.choice([-1,1])
    z = np.random.random()*2* np.random.choice([-1,1])
    return x,y,z

# performs velocity verlet
# note positions and velocities are stored within each particle in a list of particles while force is a matrix of fx,fy,fz for each particle
# instead of updating the forces, check the potential
def Verlet(particles, potentials, bc, buffer,zeta,T_inst,t,tracker):
    atom_count = len(particles)
    moved = False
    sqrdDist = 0
    trackSumrF = 0
    lst = np.arange(0,n,1)
    atoms = random.shuffle(lst)
    # this is the total potential before perturbation - potentials is a list of U for each atom
    for atom in range(0, atom_count):
        p = particles[atom]
        pold = potentials[p.id] # this will give me the potential for that particle
        x,y,z = perturb()
        p.x += x
        p.y += y
        p.z += z
        # calculate the change in potential if this were to be applied
        rFsingle,potnew = calculateForcesAndPotentials(particles,p) # this should give me the r dot F of a single particle with all other particles
        #print("rFsingle", rFsingle)
        # old and new potential for that particle -         
        pnew = potnew
        # change the list of potentials
        select = np.random.uniform(0,1)
        if select < np.exp(-(1/(kb*T)*(pnew-pold))):
            if pnew > 5:
                print("perror", pnew, atom)
            potentials[p.id] = pnew # potentials should have length 
        else:
            # put the particle back
            p.x = p.x - x
            p.y = p.y - y
            p.z = p.z - z
        # if plotDisplacement == True:
        #     p.xunroll = p.xunroll + delta * p.vx
        #     p.yunroll = p.yunroll + delta * p.vy
        #     p.zunroll = p.zunroll + delta * p.vz
        #     dx = p.xunroll - data[atom][0]
        #     dy = p.yunroll - data[atom][1]
        #     dz = p.zunroll - data[atom][2]
        #     sqrdDist += (dx**2+dy**2+dz**2)
        if boundingBox == True:
            if p.x < bc[0]:  
                p.x = p.x + L#+buffer #bc[1] minus the distance that's gone past
            elif p.x >= bc[1]:
                p.x = p.x - L#-buffer #bc[0] plus the distance past bc[1]
            if p.y >= bc[2]:
                p.y = p.y - L#-buffer #bc[3] plus the distance past bc[2]
            elif p.y < bc[3]:
                p.y = p.y + L# + buffer # bc[2] minus the distance past bc[3]
            if p.z >= bc[4]:
                p.z = p.z - L #- buffer
            elif p.z < bc[5]:
                p.z = p.z + L #+buffer       
    f_out, rFsum,potentials = calculateAllForcesAndPotentials(particles) # this should give the sum for all particles of r dot F ( after adjustment)
    K,px,py,pz,T_inst = kinetic(particles)
    if tracker == True: # keep a log of positions, velocities, and forces for each time step
        p.t = t
        p.addPoints(f_out[atom])
    # zeta = zeta + delta * 1/(T_damp**2)*(T_inst/T - 1)
    # If zeta is negative, velocity should increase, if it is positive, should decrease
    # for atom in range(0, atom_count):
    #     p = particles[atom]
    #     if noseHoover:
    #         p.vx = (p.vx + (delta/2)*f_out[p.id][0]/m)/(1+delta/2*zeta)
    #         p.vy = (p.vy + (delta/2)*f_out[p.id][0]/m)/(1+delta/2*zeta)
    #         p.vz = (p.vz + (delta/2)*f_out[p.id][0]/m)/(1+delta/2*zeta)
    #     else:
    #         p.vx = p.vx + (delta / 2) * (f_out[atom,0] / m)
    #         p.vy = p.vy + (delta/2) * (f_out[atom,1]/m)
    #         p.vz = p.vz + (delta/2) * (f_out[atom,2]/m)
                # I am using my bounding box as a simulation cell
    #K,px,py,pz,T_inst = kinetic(particles)
         # total kinetic of system
    # print("T_inst", T_inst)
    # print("T_inst/T", T_inst/T)
    # print("zeta", zeta)
    return f_out,potentials,rFsum,K,px,py,pz,T_inst,zeta,sqrdDist

# Calculates Force and Potential Energy using Leonard-Jones Potential for a single atom
# sigma - distance between atoms
# e - energy 
# r = distance between neighboring atoms

########################################  This code runs the simulation - controls everything with time #####################################################

# Moves the particles in whatever direction they are forced
def performVerlet(particles,steps,tracker,continuousFCutoff,bc,V,maxV,N,L):
    buffer = maxV*delta
    zeta = 0
    all_forces = 0

    inst_Temp = []
    inst_Press = []
    all_kinetic = []
    momentum = []
    all_potentials = []
    F,rFsum,potentials= calculateAllForcesAndPotentials(particles)
    U = np.sum(potentials)
    avgU = U
    avgP = 0
    K,_,_,_,T_inst = kinetic(particles)
    # print("\nstep no: ", 0)
    # print("kinetic: ", K, "Temp", T_inst)
    # print("density: ",n*m/V )
    # print("Pressure: ", (n*kb*T_inst/L**3+rFsum*1/(3*L**3)), "T: ", T_inst)
    density = []
    meanSq = []
    #for each step in time
    t = 0
    avgU_lst = []
    avgP_lst = []
    for i in range(steps):    
    #performs Verlet for each particle in x, y, z directions
        # if i > Equilibrium:
        #     if L > (3 * 10^(-6)):
        #         if L > 1:
        #             L -= .05
        #         elif L<= 1.01 and L > 0.0001:
        #             L -= 0.01
        #         else:
        #             L -= 0.0000000001
        #         print("L",L)
        #         if L < 0:
        #             print("help!", i)
        #     bc = createBoundaryCond[COM,L]
        if i % 100 == 0:
            print("step", i)
        density += [n/(L**3)]
        t = i*delta
        #K,px,py,pz,T_inst = kinetic(particles) # total kinetic of system  
        F,potentials,rFsum,K,px,py,pz,T_inst,zeta,sqrdDist= Verlet(particles,potentials,bc,buffer,zeta,T_inst,t,tracker)
        meanSq += [sqrdDist/n]
        all_forces+= F
        # print("U", U)
        if i > 50:
            U = np.sum(potentials)
            all_potentials += [U]
        all_kinetic += [K]
        inst_Temp += [T_inst]
        momentum += [[px,py,pz]]
        press = n*kb*T_inst/L**3+rFsum*1/(3*L**3)
        # print("ideal gas", n*kb*T_inst/L**3)
        inst_Press += [press] #need to add r dot F
        # compute averages
        if i > 50:
            avgU = (avgU*(i+1)+U)/(i+2)
            avgU_lst += [avgU]
            avgP = (avgP*(i+1)+press)/(i+2)
            avgP_lst += [avgP]

        generatetxt(i,particles)
        N = n
        for i in range(len(particles)):
            p = particles[i]
            # I am using my bounding box as a simulation cell
            if tracker == True: # keep a log of positions, velocities, and forces for each time step
                p.t = t
                p.addPoints(F[i])
        # if i > (steps - 100):
        #     ("\nstep no: ", i+1)
        #     print("kinetic: ", K, "Temp", T_inst)
        #     print("density: ",n*m/V )
        #     print("Pressure: ", (n*kb*T_inst/L**3+rFsum*1/(3*L**3)))
        #     print("Potential: ", np.sum(U))
    #print("density", density)
    return all_forces, all_potentials, all_kinetic, momentum, inst_Press, inst_Temp,density,meanSq, avgU_lst, avgP_lst

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

def generatetxt(i,particles):
    fname = 'C:/Users/madel/Documents/Mcgaughey/particle_def/10_test.txt' #' + str(i) + "
    f = open(fname,"a")
    f.write("%i\n\n"%n)
    for p in particles:
        f.write("Ar")
        f.write("\t%4.3f"%p.x)
        f.write("\t%4.3f"%p.y)
        f.write("\t%4.3f\n"%p.z)
    f.close()
    return 

def plotEnergies(particles,Energies):
    p = particles[0]
    times = p.ttrace
    plt.figure()
    # print("potential: ",Energies[0])
    # print("kinetic: ", Energies[1])
    plt.plot(times,Energies[0],label="potential")
    plt.plot(times,Energies[1],label="kinetic")
    plt.plot(times,Energies[2],label="hamiltonian")
    plt.legend()
    plt.title("Energy over time")
    plt.show()

# plot of momentum over time given particle velocities
def plotMomentum(momentums):
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

def plotTemperature(inst_Temp):
    p = particles[0]
    times = p.ttrace
    plt.figure()
    plt.plot(times,inst_Temp,label = "temperature")
    plt.title("Instantaneous Temperature")
    plt.show()

def plotPressure(inst_Press):
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

def maxwell_boltzmann(v, m, k, T):
    return 4*np.pi*(m/(2*np.pi*k*T))**(1/2)*v**2*np.exp(-m*v**2/(2*k*T)) # before I had this for speed, not velocity. This will also be the same as if I used KE for the energy term.

def plotMaxBolt(particles):
    initial_parameters = [m, kb, T]
    bins = 20
    times = particles[0].ttrace
    vsum = []
    MB = []
    for i in range(len(times) - 100,len(times)-1):
        for p in particles:
            v = np.sqrt(p.vxtrace[i]**2 + p.vytrace[i]**2 + p.vztrace[i]**2)
            vsum += [v]
            MB += [maxwell_boltzmann(v, m, kb, T)]
    plt.figure()
    plt.title("Maxwell Boltzman/ Velocity fit for 100 time steps")
    plt.hist(vsum, bins=50, density=True, alpha=0.6, color='b', label='Velocity Data')
    plt.scatter(vsum,MB,c="r", label='Fitted Maxwell-Boltzmann Distribution')
    plt.legend()
    plt.xlabel('Velocity')
    # plt.plot(times, maxwell.pdf(times, *params), lw=3)
    plt.xlabel("Velocity")
    #plt.hist(vsum/(100*n), bins=20)
    plt.show()

def plotMeanSq(meanSq):
    times = np.arange(0,steps*delta,delta)

    errors = np.array(meanSq)
    coeff = np.polyfit(times,errors,1)
    print("coefficients: ", coeff)
    print("Self Diffusion Coefficient: ", coeff[0]/6)

    plt.figure()
    plt.title("Mean Squared Displacement")
    plt.xlabel("times (nondimensional)")
    plt.ylabel("mean displacement (nondimensional)")
    plt.plot(times,meanSq,color = "b",label = "mean squared displacement")
    plt.plot(times,coeff[0]*times + coeff[1],color = "r", label = "best fit")
    plt.legend()
    plt.show()
    #Go through each particle and create a matrix of the initial position of each (this is basically data)
    #Go through each particle in particles and calculate the mean squared displacement with time

def plotAvg(particles,avgP_lst,avgU_lst):
    t = np.arange(0,len(avgP_lst))
    print("lenPlst", len(avgP_lst))
    plt.figure()
    plt.subplot(211)
    plt.plot(t,avgP_lst)
    plt.title(" Expected Pressure")

    plt.subplot(212)
    plt.plot(t,avgU_lst)
    plt.title("Expected Potential")
    plt.show()

# def plotPressDensity(inst_Press,density):
#     plt.figure()
#     plt.title("Simulated Pressure versus Density at 100K")
#     plt.xlabel("Density (kg/m^3)")
#     plt.ylabel("Pressure (N/m^2)")
#     plt.plot(density,inst_Press[0:500])
#     plt.show()

    
    # maxwell = stats.maxwell       
    # params = maxwell.fit(vsum, floc=0)

    # hist, bins = np.histogram(vsum, bins=50, density=True)
    # # Get the bin centers
    # bin_centers = (bins[:-1] + bins[1:]) / 2


    # # Define initial parameter estimates for the fit
    # initial_parameters = [m, kb, T]  # m, 
    # params, covariance = curve_fit(maxwell_boltzmann, bin_centers, hist, p0=initial_parameters)

    # fitted_mass,fitted_k,fitted_temperature = params
    # v = np.linspace(0, np.max(vsum), 100)
    # fitted_distribution = maxwell_boltzmann(v, fitted_mass, fitted_k, fitted_temperature)
    # plt.figure()
    # plt.title("Maxwell Boltzman/ Velocity fit for 100 time steps")
    # plt.hist(vsum, bins=50, density=True, alpha=0.6, color='b', label='Velocity Data')
    # plt.scatter(vsum,MB,c="r", label='Fitted Maxwell-Boltzmann Distribution')
    # plt.legend()
    # plt.xlabel('Velocity')
    # # plt.plot(times, maxwell.pdf(times, *params), lw=3)
    # plt.xlabel("Velocity")
    # #plt.hist(vsum/(100*n), bins=20)
    # plt.show()
    
################################################### Run the Code ###########################################################
if boundingBox == True:
    COM = calcCOM(x,y,z)
    bc = createBoundaryCond(COM,L)
else:
    bc = None
vi_x,vi_y,vi_z,maxV = randMomentum(n,T)
#vi_x,vi_y,vi_z,maxV = init_vel()
particles = createParticles(x,y,z,vi_x,vi_y,vi_z)
forces,potential,kineticE,momentum,inst_Press, inst_Temp,density,meanSq,avgU_lst,avgP_lst = performVerlet(particles,steps,True,continuousFCutoff,bc,V,maxV,n,L)
# H = calculateHamiltonian(kineticE,potential)
# Energies = [potential,kineticE,H]
# plotEnergies(particles,Energies)
# plotMomentum(momentum)
# plotTemperature(inst_Temp)
# plotPressure(inst_Press)
# tabulateQuant(potential,kineticE,inst_Press,inst_Temp)
#plotMeanSq(meanSq)
plotAvg(particles,avgP_lst,avgU_lst)
EqTime = isEquilibrated(inst_Temp,inst_Press)
print("Eqtime: ", EqTime)
# var = computeVariance(H[(len(H)-200):len(H)])
# print("Variance of Energy in the last 200 steps", var)

if steps > 100:
    plotMaxBolt(particles)
# if steps > Equilibrium:
#     plotPressDensity(inst_Press,density)

COM = calcCOM(x,y,z)

# I want to show convergence for the value of avgU_lst and avgP_lst (skipping)
def computeVariance(data):
    mean_sig = np.mean(data)
    var = 0
    for i in range(len(data)):
        var += (data[i] - mean_sig)**2
    return var/len(data)

def isEquilibrated(val,window_size=100):
    for i in range(int(len(val)-window_size)):
        data = val[i:int((i+window_size))]
        var = computeVariance(data)
        if var < 0.001:
            print("Equilibrated at step: ",i+window_size/2)
            return i
    return False

isEquilibrated(avgU_lst)
isEquilibrated(avgP_lst)






