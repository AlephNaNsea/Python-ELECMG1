import numpy as np
import matplotlib.pyplot as plt
import time

# Grid Size Setup
N = [50, 100, 200, 400, 800]
timings = []

# Grid Size Loop
for n in N:
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    # Vector field V = (x, -y)
    Vx = X
    Vy = -Y

    # Timings of the gradients
    start_time = time.time()
    dVx_dx, dVx_dy = np.gradient(Vx, dx, dy)
    dVy_dx, dVy_dy = np.gradient(Vy, dx, dy)
    elapsed_time = time.time() - start_time
    timings.append(elapsed_time)

# Plot highest resolution
x = np.linspace(-2, 2, N[-1])
y = np.linspace(-2, 2, N[-1])
X, Y = np.meshgrid(x, y)
Vx = X
Vy = -Y

plt.figure(figsize=(8, 6))
plt.quiver(X[::20, ::20], Y[::20, ::20], Vx[::20, ::20], Vy[::20, ::20], color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Vector Field: V(x,y) = (x, -y)')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Performance vs Grid Size
grid_points = [n**2 for n in N]

plt.figure(figsize=(8, 6))
plt.loglog(grid_points, timings, 'o-', linewidth=2)
plt.xlabel('Number of Grid Points (N²)')
plt.ylabel('Elapsed Time (s)')
plt.title('2D Vector Field: Time vs Grid Size (N²)')
plt.grid(True, which="both", linestyle='--')
plt.tight_layout()
plt.show()

# Final Grid Spacing and Nyquist Limit
dx = x[1] - x[0]
nyquist_limit = 2 * dx
gradient_error = dx ** 2

# Summary Report
print("\n--- 2D Vector Field Simulation Report ---")
print(f"Final Grid Size (N x N): {N[-1]} x {N[-1]}")
print(f"Domain Range: x, y ∈ [-2, 2]")
print(f"Grid Spacing (Δx = Δy): {dx:.6f}")
print(f"Nyquist Limit (2Δx): {nyquist_limit:.6f}")
print(f"Gradient Error Estimate (Δx²): {gradient_error:.2e}")
print(f"\nElapsed Times:")
for i in range(len(N)):
    print(f"  N = {N[i]:4d} | Grid Points = {N[i]**2:7,d} | Time = {timings[i]:.5f} sec")
print("----------------------------------------")
