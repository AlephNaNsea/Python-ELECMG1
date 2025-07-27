import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Grid Sizes
N_values = [20, 30, 40, 50]
nt = 1000
eps = mu = 1.0

timings = []
points = []

# Loop through grid sizes
for N in N_values:
    nx = ny = nz = N
    dx = dy = dz = 1.0 / (N - 1)
    dt = 0.005
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    Ex = np.zeros((nx, ny, nz))
    Ey = np.zeros((nx, ny, nz))
    Ez = np.zeros((nx, ny, nz))
    Hx = np.zeros((nx, ny, nz))
    Hy = np.zeros((nx, ny, nz))
    Hz = np.zeros((nx, ny, nz))

    t0 = time.time()
    for t in range(nt):
        # Update magnetic fields
        Hx[:, :-1, :-1] += (dt / mu) * (
            (Ey[:, :-1, 1:] - Ey[:, :-1, :-1]) / dz -
            (Ez[:, 1:, :-1] - Ez[:, :-1, :-1]) / dy
        )
        Hy[:-1, :, :-1] += (dt / mu) * (
            (Ez[1:, :, :-1] - Ez[:-1, :, :-1]) / dx -
            (Ex[:-1, :, 1:] - Ex[:-1, :, :-1]) / dz
        )
        Hz[:-1, :-1, :] += (dt / mu) * (
            (Ex[:-1, 1:, :] - Ex[:-1, :-1, :]) / dy -
            (Ey[1:, :-1, :] - Ey[:-1, :-1, :]) / dx
        )

        # Update electric fields
        Ex[1:, 1:, 1:] += (dt / eps) * (
            (Hz[1:, 1:, 1:] - Hz[1:, :-1, 1:]) / dy -
            (Hy[1:, 1:, 1:] - Hy[1:, 1:, :-1]) / dz
        )
        Ey[1:, 1:, 1:] += (dt / eps) * (
            (Hx[1:, 1:, 1:] - Hx[1:, 1:, :-1]) / dz -
            (Hz[1:, 1:, 1:] - Hz[:-1, 1:, 1:]) / dx
        )
        Ez[1:, 1:, 1:] += (dt / eps) * (
            (Hy[1:, 1:, 1:] - Hy[:-1, 1:, 1:]) / dx -
            (Hx[1:, 1:, 1:] - Hx[1:, :-1, 1:]) / dy
        )

        # Inject sinusoidal source
        Ex[cx, cy, cz] += 20 * np.sin(2 * np.pi * 50 * t * dt) * np.exp(-t ** 2 / 200)

    elapsed = time.time() - t0
    timings.append(elapsed)
    points.append(N ** 3)
    print(f"Simulation and frame capture completed in {elapsed:.2f} seconds.")

# Performance Plot
plt.figure(figsize=(8, 6))
plt.loglog(points, timings, 'o-', linewidth=2)
plt.xlabel('Number of Grid Points (N³)')
plt.ylabel('Elapsed Time (s)')
plt.title('3D FDTD Performance Benchmark')
plt.grid(True, which="both", linestyle='--')
plt.tight_layout()
plt.show()

# Simulation (Largest N)
final_N = N_values[-1]
nx = ny = nz = final_N
dx = dy = dz = 1.0 / (final_N - 1)
dt = 0.005
skip = 5

Ex = np.zeros((nx, ny, nz))
Ey = np.zeros((nx, ny, nz))
Ez = np.zeros((nx, ny, nz))
Hx = np.zeros((nx, ny, nz))
Hy = np.zeros((nx, ny, nz))
Hz = np.zeros((nx, ny, nz))

cx, cy, cz = nx // 2, ny // 2, nz // 2

x = np.arange(nx)
y = np.arange(ny)
z = np.arange(nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
Xq, Yq, Zq = X[::skip, ::skip, ::skip], Y[::skip, ::skip, ::skip], Z[::skip, ::skip, ::skip]

origin_x = Xq - nx // 2
origin_y = Yq - ny // 2
origin_z = Zq - nz // 2
origin_norm = np.sqrt(origin_x**2 + origin_y**2 + origin_z**2)
origin_norm[origin_norm == 0] = 1

frames = []

print("Simulating and capturing frames...")
start_time = time.time()

for t in range(nt):
    Hx[:, :-1, :-1] += (dt / mu) * (
        (Ey[:, :-1, 1:] - Ey[:, :-1, :-1]) / dz -
        (Ez[:, 1:, :-1] - Ez[:, :-1, :-1]) / dy
    )
    Hy[:-1, :, :-1] += (dt / mu) * (
        (Ez[1:, :, :-1] - Ez[:-1, :, :-1]) / dx -
        (Ex[:-1, :, 1:] - Ex[:-1, :, :-1]) / dz
    )
    Hz[:-1, :-1, :] += (dt / mu) * (
        (Ex[:-1, 1:, :] - Ex[:-1, :-1, :]) / dy -
        (Ey[1:, :-1, :] - Ey[:-1, :-1, :]) / dx
    )

    Ex[1:, 1:, 1:] += (dt / eps) * (
        (Hz[1:, 1:, 1:] - Hz[1:, :-1, 1:]) / dy -
        (Hy[1:, 1:, 1:] - Hy[1:, 1:, :-1]) / dz
    )
    Ey[1:, 1:, 1:] += (dt / eps) * (
        (Hx[1:, 1:, 1:] - Hx[1:, 1:, :-1]) / dz -
        (Hz[1:, 1:, 1:] - Hz[:-1, 1:, 1:]) / dx
    )
    Ez[1:, 1:, 1:] += (dt / eps) * (
        (Hy[1:, 1:, 1:] - Hy[:-1, 1:, 1:]) / dx -
        (Hx[1:, 1:, 1:] - Hx[1:, :-1, 1:]) / dy
    )

    Ex[cx, cy, cz] += 20 * np.sin(2 * np.pi * 50 * t * dt) * np.exp(-t ** 2 / 200)

    U = Ex[::skip, ::skip, ::skip]
    V = Ey[::skip, ::skip, ::skip]
    W = Ez[::skip, ::skip, ::skip]

    mag = np.sqrt(U**2 + V**2 + W**2)
    mag[mag == 0] = 1

    dot = U * origin_x + V * origin_y + W * origin_z
    cos_sim = np.clip(dot / (mag * origin_norm), -1, 1)
    color_array = plt.cm.seismic(((cos_sim + 1) / 2).flatten())
    frames.append((U.flatten(), V.flatten(), W.flatten(), color_array))

elapsed_total = time.time() - start_time
print(f"Simulation and frame capture completed in {elapsed_total:.2f} seconds.")

# Animation
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
Xf = Xq.flatten()
Yf = Yq.flatten()
Zf = Zq.flatten()

def update_quiver(frame_idx):
    ax.cla()
    ax.set_xlim([0, nx])
    ax.set_ylim([0, ny])
    ax.set_zlim([0, nz])
    ax.set_title(f"Shaded E Field at N = 50– Frame {frame_idx}")
    U, V, W, colors = frames[frame_idx]
    ax.quiver(Xf, Yf, Zf, U, V, W, length=1.5, normalize=True, colors=colors)
    return ax,

ani = animation.FuncAnimation(fig, update_quiver, frames=nt, interval=1, blit=False)
ani.save("fdtd_e_field_animation.mp4", fps=100, dpi=150)
plt.show()

# Summary Report
print("\n--- 3D FDTD Electromagnetic Field Simulation Report ---")
print(f"Grid Size: {nx} x {ny} x {nz}")
print(f"Time Steps: {nt}")
print(f"Grid Spacing (Δx = Δy = Δz): {dx:.6f}")
print(f"Time Step (Δt): {dt:.6f}")
print(f"Skip Interval (Vector Sampling): {skip}")
print(f"Permittivity (ε): {eps}")
print(f"Permeability (μ): {mu}")
print(f"Source Frequency: 50 Hz (sinusoidal injection)")
print(f"Source Location: ({cx}, {cy}, {cz})")
print(f"Vector Shading: Based on Cosine Similarity to Radial Direction")
print(f"Animation Frames: {nt}")
print(f"Vector Field Visualized: Electric Field (E) [Ex, Ey, Ez]")
print(f"Elapsed Time: {elapsed_total:.2f} seconds")
