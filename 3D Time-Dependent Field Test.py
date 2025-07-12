import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import time

# Parameters
nx = ny = nz = 50
nt = 200
alpha = 0.01
L = 1.0
dx = dy = dz = L / (nx - 1)
dt = 0.005

# Grid
x = np.linspace(0, L, nx)
y = np.linspace(0, L, ny)
z = np.linspace(0, L, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Initial condition: centered Gaussian hotspot
sigma2 = 0.01
T = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2) / sigma2)
start_time = time.time()

# Time evolution (Euler)
for _ in range(nt):
    laplacian = (
        np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) +
        np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) +
        np.roll(T, 1, axis=2) + np.roll(T, -1, axis=2) - 6 * T
    ) / dx**2
    T += alpha * dt * laplacian

elapsed = time.time() - start_time
print(f"Simulation completed in {elapsed:.2f} seconds.")

# Isosurface Visualization
T_min, T_max = T.min(), T.max()
level = 0.5 * (T_min + T_max)
print(f"T min: {T_min:.4f}, T max: {T_max:.4f}")
print(f"Generating isosurface at T = {level:.3f}...")

verts, faces, _, _ = measure.marching_cubes(T, level=level, spacing=(dx, dy, dz))

fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='hot', lw=0.2)
ax1.set_title(f'Isosurface of Temperature ≈ {level:.2f} at t = {nt * dt:.3f} s')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
plt.tight_layout()
plt.show()

# --- 3D Volume Slice Plot ---
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection='3d')

# Downsample for clearer slicing
slice_step = 5
T_sliced = T[::slice_step, ::slice_step, ::slice_step]
x_s = x[::slice_step]
y_s = y[::slice_step]
z_s = z[::slice_step]
X_s, Y_s, Z_s = np.meshgrid(x_s, y_s, z_s, indexing='ij')

scat = ax2.scatter(X_s, Y_s, Z_s, c=T_sliced.flatten(), cmap='plasma', alpha=0.9, marker='s', s=15)
fig2.colorbar(scat, ax=ax2, shrink=0.6, label='Temperature')
ax2.set_title(f'T at t = {nt * dt:.3f} s')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
plt.tight_layout()
plt.show()

# Summary Report
print("\n--- 3D Heat Diffusion Report ---")
print(f"Grid Size: {nx} x {ny} x {nz}")
print(f"Time Steps: {nt}")
print(f"Grid Spacing (Δx = Δy = Δz): {dx:.6f}")
print(f"Time Step (Δt): {dt:.6f}")
print(f"Diffusivity (α): {alpha}")
print(f"Elapsed Time: {elapsed:.2f} seconds")
print(f"Nyquist Limit (2Δx): {2 * dx:.6f}")
print(f"Gradient Error Estimate (Δx²): {dx**2:.2e}")
print("--------------------------------")
