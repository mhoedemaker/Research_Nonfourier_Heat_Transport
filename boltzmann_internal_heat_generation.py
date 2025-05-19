import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1e-6             # Slab length [m]
Nx = 100             # Number of spatial points
Nl = 16              # Number of angular directions
v = 3000             # Group velocity [m/s]
tau = 1e-10          # Relaxation time [s]
ell = v * tau        # Mean free path
C = 1.6e6            # Heat capacity per unit volume [J/m³·K]
Q0 = 1e9             # Uniform heat generation [W/m³]
T_L = 310            # Left boundary temperature [K]
T_R = 300            # Right boundary temperature [K]

# Discretization
x = np.linspace(0, L, Nx)
dx = x[1] - x[0]
mu, w = np.polynomial.legendre.leggauss(Nl)
f = np.zeros((Nl, Nx))
f0 = np.linspace(T_L, T_R, Nx)  # Initial temperature guess
Q = np.full(Nx, Q0)             # Heat generation array

# Boundary conditions
for j in range(Nl):
    if mu[j] > 0:
        f[j, 0] = T_L
    else:
        f[j, -1] = T_R

# Iterative solution
for _ in range(300):
    f_old = f0.copy()
    # Sweep over all angles
    for j in range(Nl):
        if mu[j] > 0:
            for i in range(1, Nx):
                numer = f[j, i-1] + dx * f0[i] / (ell * mu[j])
                denom = 1 + dx / (ell * mu[j])
                f[j, i] = numer / denom
        else:
            for i in reversed(range(Nx - 1)):
                numer = f[j, i+1] + dx * f0[i] / (-ell * mu[j])
                denom = 1 + dx / (-ell * mu[j])
                f[j, i] = numer / denom

    # Update local equilibrium (temperature)
    f0 = 0.5 * np.sum(w[:, None] * f, axis=0)
    # Add internal heat generation term
    f0 += (tau / C) * Q

    # Check convergence
    if np.max(np.abs(f0 - f_old)) < 1e-4:
        break

# Plot
plt.plot(x * 1e6, f0, label="BTE with Heat Generation")
plt.xlabel("x [μm]")
plt.ylabel("Temperature [K]")
plt.title("1D BTE with Internal Heat Generation")
plt.grid(True)
plt.legend()
plt.show()