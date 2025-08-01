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
grid_points = [n**2 for n in N]
plt.figure(figsize=(8, 6))
plt.loglog(grid_points, timings, 'o-', linewidth=2)
plt.xlabel('Number of Grid Points (N²)')
plt.ylabel('Elapsed Time (s)')
plt.title('2D Scalar Field: Time vs Grid Size (log-log)')
plt.grid(True, which="both", ls='--')
plt.tight_layout()
plt.show()

# Final Grid Spacing and Nyquist Limit
dx = x[1] - x[0]
nyquist_limit = 2 * dx
gradient_error = dx**2

# --- Summary Report ---
print("\n--- 2D Scalar Field Simulation Report ---")
print(f"Final Grid Size (N x N): {N[-1]} x {N[-1]}")
print(f"Domain Range: x, y ∈ [-2, 2]")
print(f"Grid Spacing (Δx = Δy): {dx:.6f}")
print(f"Nyquist Limit (2Δx): {nyquist_limit:.6f}")
print(f"Gradient Error Estimate (Δx²): {gradient_error:.2e}")
print(f"\nElapsed Times:")
for i in range(len(N)):
    print(f"  N = {N[i]:4d} | Grid Points = {N[i]**2:7,d} | Time = {timings[i]:.5f} sec")
print("----------------------------------------")