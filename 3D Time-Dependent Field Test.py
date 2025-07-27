import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import matplotlib.animation as animation

# Simulation Parameters
nt = 1000
alpha = 0.01
L = 1.0
dt = 0.005
sigma2 = 0.01

# Grid Sizes
grid_sizes = [20, 30, 40, 50]
timings = []
points = []

for n in grid_sizes:
    print(f"\nRunning Simulation for Grid Size: {n}³")
    dx = L / (n - 1)
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    z = np.linspace(0, L, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Initial condition: centered Gaussian hotspot
    T = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2) / sigma2)

    # Store frames
    isosurface_frames = []

    # Timing
    print("Simulating and capturing frames...")
    start_time = time.time()

    for step in range(nt):
        # Euler step
        laplacian = (
            np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) +
            np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) +
            np.roll(T, 1, axis=2) + np.roll(T, -1, axis=2) - 6 * T
        ) / dx**2
        T += alpha * dt * laplacian

        # Isosurface capture
        level = 0.5 * (T.min() + T.max())
        verts, faces, _, _ = measure.marching_cubes(T, level=level, spacing=(dx, dx, dx))
        isosurface_frames.append((verts, faces, level))

    elapsed = time.time() - start_time
    timings.append(elapsed)
    points.append(n**3)

    print(f"Simulation and frame capture completed in {elapsed:.2f} seconds.")

    # Create animation (for largest N only)
    if n == grid_sizes[-1]:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame_idx):
            ax.cla()
            verts, faces, level = isosurface_frames[frame_idx]
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                            cmap='hot', lw=0.2)
            ax.set_xlim([0, L])
            ax.set_ylim([0, L])
            ax.set_zlim([0, L])
            ax.set_title(f'Isosurface at t = {frame_idx * dt:.3f}s (T ≈ {level:.3f})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            return ax,

        ani = animation.FuncAnimation(fig, update, frames=nt, interval=50, blit=False)
        ani.save("heat_diffusion_isosurface.mp4", fps=100, dpi=150)
        plt.tight_layout()
        plt.show()

        # Summary Report
        print("\n--- 3D Heat Diffusion Animation Report ---")
        print(f"Grid Size: {n} x {n} x {n}")
        print(f"Time Steps: {nt}")
        print(f"Grid Spacing (Δx = Δy = Δz): {dx:.6f}")
        print(f"Time Step (Δt): {dt:.6f}")
        print(f"Diffusivity (α): {alpha}")
        print(f"Elapsed Time: {elapsed:.2f} seconds")
        print(f"Nyquist Limit (2Δx): {2 * dx:.6f}")
        print(f"Gradient Error Estimate (Δx²): {dx**2:.2e}")
        print(f"Isosurface Animation Saved As: 'heat_diffusion_isosurface.mp4'")

# Performance plot
plt.figure(figsize=(8, 6))
plt.loglog(points, timings, 'o-', linewidth=2)
plt.xlabel('Number of Grid Points (N³)')
plt.ylabel('Elapsed Time (s)')
plt.title('3D Heat Diffusion: Time vs Grid Size')
plt.grid(True, which="both", linestyle='--')
plt.tight_layout()
plt.show()

