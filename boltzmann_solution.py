import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1e-6               # Length of the slab [m]
Nx = 100               # Spatial grid points
Nl = 16                # Angular directions (even)
v = 3000               # Group velocity [m/s]
tau = 1e-10            # Relaxation time [s]
ell = v * tau          # Mean free path
T_L = 310              # Left temperature [K]
T_R = 300              # Right temperature [K]

# Discretization
x = np.linspace(0, L, Nx)
dx = x[1] - x[0]

mu, w = np.polynomial.legendre.leggauss(Nl)  # Gauss quadrature
f = np.zeros((Nl, Nx))                       # Distribution function
f0 = np.linspace(T_L, T_R, Nx)               # Initial guess for T(x)

# Boundary conditions (injected phonons)
for j in range(Nl):
    if mu[j] > 0:
        f[j, 0] = T_L
    else:
        f[j, -1] = T_R

# Iterative solution
for _ in range(200):
    f_old = f0.copy()
    # Sweep all angles
    for j in range(Nl):
        if mu[j] > 0:
            for i in range(1, Nx):
                f[j, i] = (f[j, i-1] + dx / (ell * mu[j]) * f0[i]) / (1 + dx / (ell * mu[j]))
        else:
            for i in reversed(range(Nx - 1)):
                f[j, i] = (f[j, i+1] + dx / (-ell * mu[j]) * f0[i]) / (1 + dx / (-ell * mu[j]))
    # Update local equilibrium
    f0 = 0.5 * np.sum(w[:, None] * f, axis=0)
    if np.max(np.abs(f0 - f_old)) < 1e-4:
        break

# Plot results
plt.plot(x * 1e6, f0, label="BTE solution")
plt.plot(x * 1e6, np.linspace(T_L, T_R, Nx), '--', label="Fourier (linear)")
plt.xlabel("x [μm]")
plt.ylabel("Temperature [K]")
plt.legend()
plt.title("Temperature Profile in Kinetic Regime")
plt.grid(True)
plt.show()