import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

def poincare_mapping(rgb):
    """Maps RGB values to quaternion using Poincaré mapping"""
    # Normalize RGB vector
    norm = np.linalg.norm(rgb, axis=1, keepdims=True)
    rgb_normalized = rgb / (norm + 1)
    
    # Calculate real component (q0)
    q0 = np.sqrt(1 - np.sum(rgb_normalized**2, axis=1, keepdims=True))
    
    # Return quaternion [q0, q1, q2, q3]
    return np.hstack([q0, rgb_normalized])

# Create a grid of RGB values
n = 10  # number of samples per dimension
r = np.linspace(0, 1, n)
g = np.linspace(0, 1, n)
b = np.linspace(0, 1, n)

# Create a mesh grid
R, G, B = np.meshgrid(r, g, b)
rgb_values = np.column_stack([R.flatten(), G.flatten(), B.flatten()])

# Map RGB values to quaternions using Poincaré mapping
quaternions = poincare_mapping(rgb_values)

# Create a 3D figure
fig = plt.figure(figsize=(15, 10))

# Plot 1: RGB space (unit cube)
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(
    rgb_values[:, 0], 
    rgb_values[:, 1], 
    rgb_values[:, 2],
    c=rgb_values, 
    marker='o'
)
ax1.set_xlabel('R')
ax1.set_ylabel('G')
ax1.set_zlabel('B')
ax1.set_title('RGB Color Space')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_zlim(0, 1)

# Plot 2: Fixed q0 slice of quaternion space (q1, q2, q3)
ax2 = fig.add_subplot(122, projection='3d')

# Color points based on their q0 value for visualization
q0_colors = plt.cm.viridis(quaternions[:, 0])

scatter2 = ax2.scatter(
    quaternions[:, 1],  # q1 (from R)
    quaternions[:, 2],  # q2 (from G)
    quaternions[:, 3],  # q3 (from B)
    c=quaternions[:, 0],  # Color by q0 value
    cmap='viridis',
    marker='o'
)

# Add a color bar to show q0 values
cbar = plt.colorbar(scatter2, ax=ax2, label='q0 value')

ax2.set_xlabel('q1 (from R)')
ax2.set_ylabel('q2 (from G)')
ax2.set_zlabel('q3 (from B)')
ax2.set_title('Poincaré Mapping: Quaternion Space (q1, q2, q3)')

# Add spherical boundary to show unit quaternion constraint
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax2.plot_surface(x, y, z, color='gray', alpha=0.1)

plt.tight_layout()
plt.savefig('poincare_mapping_visualization.png', dpi=300)
plt.show()

# Create another visualization to show how different RGB values map to different q0 values
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a more structured grid for better visualization
n = 5  # fewer points for clarity
r = np.linspace(0, 1, n)
g = np.linspace(0, 1, n)
b = np.linspace(0, 1, n)
R, G, B = np.meshgrid(r, g, b)
rgb_grid = np.column_stack([R.flatten(), G.flatten(), B.flatten()])

# Map to quaternions
quat_grid = poincare_mapping(rgb_grid)

# Plot points in RGB space, but size them according to q0 value
q0_sizes = 100 * quat_grid[:, 0]  # Scale for visibility
scatter = ax.scatter(
    rgb_grid[:, 0],
    rgb_grid[:, 1],
    rgb_grid[:, 2],
    c=rgb_grid,
    s=q0_sizes,
    marker='o'
)

# Add connecting lines to show mapping
for i in range(len(rgb_grid)):
    # Calculate the normalized direction
    norm = np.linalg.norm(rgb_grid[i])
    if norm > 0:
        direction = rgb_grid[i] / norm
        # Draw an arrow showing the "push" toward the center
        ax.quiver(
            rgb_grid[i, 0], rgb_grid[i, 1], rgb_grid[i, 2],  # start
            -direction[0] * 0.2, -direction[1] * 0.2, -direction[2] * 0.2,  # direction and length
            color='gray', alpha=0.3
        )

ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
ax.set_title('Poincaré Mapping: Effect on RGB Space (larger points have higher q0)')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

plt.tight_layout()
plt.savefig('poincare_q0_effect.png', dpi=300)
plt.show()