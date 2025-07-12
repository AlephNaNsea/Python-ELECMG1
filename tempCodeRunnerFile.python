import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

# 3D Grid Setup
N = 100
domain = (-1, 1)
x = np.linspace(*domain, N)
y = np.linspace(*domain, N)
z = np.linspace(*domain, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Grid Spacing Compute
dx = x[1] - x[0]
nyquist_limit = 2 * dx
gradient_error = dx ** 2

# Start timing
start_time = time.time()

# Define scalar field: phi = x² + y² + z²
phi = X**2 + Y**2 + Z**2

# Compute gradient (Fx, Fy, Fz)
Fx, Fy, Fz = np.gradient(phi, dx, dx, dx)

# Elapsed time
elapsed = time.time() - start_time

# Visualize isosurface phi = 0.5
verts, faces, _, _ = measure.marching_cubes(phi, level=0.5, spacing=(dx, dx, dx))

# Isosurface Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=0.2)
ax.set_title("3D Scalar Field Isosurface (φ = 0.5)")
plt.tight_layout()
plt.show()

# Summary Report
print("\n--- 3D Scalar Field Simulation Report ---")
print(f"Grid Size (N x N x N): {N} x {N} x {N}")
print(f"Grid Spacing (Δx = Δy = Δz): {dx:.4f}")
print(f"Nyquist Limit (2Δx): {nyquist_limit:.4f}")
print(f"Gradient Error Estimate (Δx²): {gradient_error:.2e}")
print(f"Elapsed Time (Field + Gradient Computation): {elapsed:.4f} seconds")
print(f"Isosurface Rendered at φ = 0.5 (Sphere Radius ≈ {np.sqrt(0.5):.4f})")
print("----------------------------------------")

# Time vs N³ Plot

N_values = [20, 40, 60, 80, 100]
times = []
points = []

for N in N_values:
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    z = np.linspace(-1, 1, N)
    dx = x[1] - x[0]

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    start = time.time()
    phi = X**2 + Y**2 + Z**2
    Fx, Fy, Fz = np.gradient(phi, dx, dx, dx)
    elapsed = time.time() - start

    times.append(elapsed)
    points.append(N**3)

    print(f"N = {N:3d} | Δx = {dx:.4f} | Time = {elapsed:.4f} sec | Grid Points = {N**3:,}")

plt.figure(figsize=(8, 6))
plt.loglog(points, times, 'o-', linewidth=2)
plt.xlabel('Number of Grid Points (N³)')
plt.ylabel('Elapsed Time (s)')
plt.title('3D Scalar Field: Time vs Grid Size (N³)')
plt.grid(True, which="both", ls='--')
plt.tight_layout()
plt.show()
