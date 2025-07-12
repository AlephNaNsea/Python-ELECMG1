import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Grid Size
N = [20, 40, 60, 80, 100]
timings = []

# Loop over each grid size
for n in N:
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    z = np.linspace(-1, 1, n)
    dx = x[1] - x[0]

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Define vector field V = (x, -y, z)
    Vx = X
    Vy = -Y
    Vz = Z

    # Time the gradient (∂Vx, ∂Vy, ∂Vz)
    start_time = time.time()
    dVx = np.gradient(Vx, dx, dx, dx)
    dVy = np.gradient(Vy, dx, dx, dx)
    dVz = np.gradient(Vz, dx, dx, dx)
    elapsed_time = time.time() - start_time
    timings.append(elapsed_time)

# 3D Vector Field Plot
N_plot = N[-1]
x = np.linspace(-1, 1, N_plot)
y = np.linspace(-1, 1, N_plot)
z = np.linspace(-1, 1, N_plot)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

Vx = X
Vy = -Y
Vz = Z

# Subsample the grid for plotting to avoid overflow
step = 6
Xq = X[::step, ::step, ::step]
Yq = Y[::step, ::step, ::step]
Zq = Z[::step, ::step, ::step]
Vxq = Vx[::step, ::step, ::step]
Vyq = Vy[::step, ::step, ::step]
Vzq = Vz[::step, ::step, ::step]

# Create 3D quiver plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(Xq, Yq, Zq, Vxq, Vyq, Vzq, length=0.1, normalize=True, color='blue')
ax.set_title('3D Vector Field: V(x,y,z) = (x, -y, z)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.tight_layout()
plt.show()

# Log-log performance plot
grid_points = [n**3 for n in N]
plt.figure(figsize=(8, 6))
plt.loglog(grid_points, timings, 'o-', linewidth=2)
plt.xlabel('Number of Grid Points (N³)')
plt.ylabel('Elapsed time (s)')
plt.title('3D Vector Field: Time vs Grid Size (log-log)')
plt.grid(True, which="both", linestyle='--')
plt.tight_layout()
plt.show()

# Final Grid Spacing and Nyquist Limit
dx = x[1] - x[0]
nyquist_limit = 2 * dx
gradient_error = dx ** 2

# Summary Report
print("\n--- 3D Vector Field Simulation Report ---")
print(f"Final Grid Size (N x N x N): {N[-1]} x {N[-1]} x {N[-1]}")
print(f"Domain Range: x, y, z ∈ [-1, 1]")
print(f"Grid Spacing (Δx = Δy = Δz): {dx:.6f}")
print(f"Nyquist Limit (2Δx): {nyquist_limit:.6f}")
print(f"Gradient Error Estimate (Δx²): {gradient_error:.2e}")
print(f"\nElapsed Times:")
for i in range(len(N)):
    print(f"  N = {N[i]:4d} | Grid Points = {N[i]**3:8,d} | Time = {timings[i]:.5f} sec")
print("----------------------------------------")
