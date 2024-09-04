# import open3d as o3d
# import numpy as np

# # Generate some random points
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))

# # Visualize
# o3d.visualization.draw_geometries([pcd])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Generate some random points
point_set = np.random.rand(100, 3)
colors = np.random.rand(100, 3)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with colors
ax.scatter(point_set[:, 0], point_set[:, 1], point_set[:, 2], c=colors)

# Show the plot
plt.show()