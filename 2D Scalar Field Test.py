import numpy as np
import matplotlib.pyplot as plt
import time

# Define grid sizes
N = [50, 100, 200, 400, 800]
timings = []

# Loop over grid sizes
for n in N:
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, y)

    # Scalar field: phi(x,y) = sin(pi x)*cos(pi y)
    phi = np.sin(np.pi * X) * np.cos(np.pi * Y)

    # Gradient Calculation Elapsed Time
    start_time = time.time()
    Fx, Fy = np.gradient(phi, x, y)
    elapsed_time = time.time() - start_time
    timings.append(elapsed_time)

# Plot highest resolution
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, phi, levels=20, cmap='viridis')
plt.colorbar(label='Scalar Field Value')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title(r'2D Scalar Field: $\phi(x,y) = \sin(\pi x) \cos(\pi y)$')
plt.show()

# Performance vs Grid Size
plt.figure(figsize=(8, 6))
plt.plot(N, timings, 'o-', linewidth=2)
plt.xlabel('Grid size N x N')
plt.ylabel('Elapsed time (s)')
plt.title('Performance vs. Grid Resolution')
plt.grid(True)
plt.show()

# Final Grid Spacing and Nyquist Limit
dx = x[1] - x[0]
print(f"Final grid spacing: dx = {dx:.6f}")
print(f"Nyquist limit: 2dx = {2*dx:.6f}")
