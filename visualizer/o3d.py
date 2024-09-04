import numpy as np
import ctypes as ct
import sys
import os
import cv2
import argparse
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append("/mnt/c/Users/kaise/Downloads/Pointnet_Pointnet2_pytorch/data_utils")

path_to_check = "/mnt/c/Users/kaise/Downloads/Pointnet_Pointnet2_pytorch/data_utils"

if os.path.exists(path_to_check):
    print(f"Path exists: {path_to_check}")
else:
    print(f"Path does not exist: {path_to_check}")

from S3DISDataLoader import S3DISDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
showsz = 800
mousex, mousey = 0.5, 0.5
zoom = 1.0

dll = np.ctypeslib.load_library(os.path.join(BASE_DIR, 'render_balls_so'), '.')

def render_to_image(xyz, c0, c1, c2, ballradius=10, background=(0, 0, 0), freezerot=False):
    global showsz, zoom
    show = np.zeros((showsz, showsz, 3), dtype='uint8')
    
    rotmat = np.eye(3)
    if not freezerot:
        xangle = (mousey - 0.5) * np.pi * 1.2
        yangle = (mousex - 0.5) * np.pi * 1.2
    else:
        xangle, yangle = 0, 0

    rotmat = rotmat.dot(np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(xangle), -np.sin(xangle)],
        [0.0, np.sin(xangle), np.cos(xangle)],
    ]))
    rotmat = rotmat.dot(np.array([
        [np.cos(yangle), 0.0, -np.sin(yangle)],
        [0.0, 1.0, 0.0],
        [np.sin(yangle), 0.0, np.cos(yangle)],
    ]))
    rotmat *= zoom

    # Extract only the first 3 columns (x, y, z) for rotation
    coords = xyz[:, :3]  # Extract the first 3 columns for rotation
    nxyz = coords.dot(rotmat) + [showsz / 2, showsz / 2, 0]  # Apply rotation

    ixyz = nxyz.astype('int32')
    show[:] = background
    dll.render_ball(
        ct.c_int(show.shape[0]),
        ct.c_int(show.shape[1]),
        show.ctypes.data_as(ct.c_void_p),
        ct.c_int(ixyz.shape[0]),
        ixyz.ctypes.data_as(ct.c_void_p),
        c0.ctypes.data_as(ct.c_void_p),
        c1.ctypes.data_as(ct.c_void_p),
        c2.ctypes.data_as(ct.c_void_p),
        ct.c_int(ballradius)
    )
    return show

def render_realtime(xyz, c_gt=None, c_pred=None, showrot=False, magnifyBlue=0, 
                    freezerot=False, background=(255, 255, 255), normalizecolor=True, ballradius=10):
    xyz = xyz - xyz.mean(axis=0)
    radius = ((xyz ** 2).sum(axis=-1) ** 0.5).max()
    xyz /= (radius * 2.2) / showsz
    if c_gt is None:
        c0 = np.zeros((len(xyz),), dtype='float32') + 255
        c1 = np.zeros((len(xyz),), dtype='float32') + 255
        c2 = np.zeros((len(xyz),), dtype='float32') + 255
    else:
        c0 = c_gt[:, 0]
        c1 = c_gt[:, 1]
        c2 = c_gt[:, 2]

    if normalizecolor:
        c0 /= (c0.max() + 1e-14) / 255.0
        c1 /= (c1.max() + 1e-14) / 255.0
        c2 /= (c2.max() + 1e-14) / 255.0

    c0 = np.require(c0, 'float32', 'C')
    c1 = np.require(c1, 'float32', 'C')
    c2 = np.require(c2, 'float32', 'C')

    img = render_to_image(xyz, c0, c1, c2, ballradius=ballradius, background=background, freezerot=freezerot)
    
    # Display the result in a window
    cv2.imshow('3D Point Cloud', img)

    # Wait for a key press to exit the visualization
    key = cv2.waitKey(1)  # Set delay in ms (1 ms for real-time)
    if key == 27:  # Press 'ESC' to exit
        cv2.destroyAllWindows()
        return False
    return True

def visualize_open3d(points, colors):
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    
    # Assign points and colors to the point cloud
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

def visualize_headless(points, colors, output_file="output_image.png"):
    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Set up an offscreen renderer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Do not create a visible window
    vis.add_geometry(pcd)

    # Render the scene to an image
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_file)  # Save the rendering to an image
    vis.destroy_window()

    print(f"Saved point cloud visualization to {output_file}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='../data/ss3dis/stanford_indoor3d', help='dataset path')
#     parser.add_argument('--category', type=str, default='Floor', help='select category')
#     parser.add_argument('--npoints', type=int, default=25000, help='resample points number')
#     parser.add_argument('--ballradius', type=int, default=1, help='ballradius')
#     opt = parser.parse_args()

#     cmap = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],  # Class 0
#                     [3.12493437e-02, 1.00000000e+00, 1.31250131e-06],  # Class 1
#                     [0.00000000e+00, 6.25019688e-02, 1.00000000e+00],  # Class 2
#                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 3
#                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 4
#                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 5
#                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 6
#                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 7
#                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 8
#                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 9
#                     [0.50000000e+00, 0.50000000e+00, 0.00000000e+00],  # Class 10
#                     [0.00000000e+00, 1.00000000e+00, 0.50000000e+00],  # Class 11
#                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]) # Class 12

#     dataset = S3DISDataset(split='train', data_root='data/stanford_indoor3d', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0)

#     idx = np.random.randint(0, len(dataset))
#     print(f"Total samples in dataset: {len(dataset)}")
#     idx = np.random.randint(0, len(dataset))
#     current_points, current_labels = dataset[idx]
#     print(f"Selected sample index: {idx}")
#     print(f"Current points shape: {current_points.shape}")
#     print(f"Current labels shape: {current_labels.shape}")
#     print(f"Datatype of current_labels: {current_labels.dtype}")
#     current_labels = current_labels.astype(np.int32)
#     print(f"Datatype of current_labels after conversion: {current_labels.dtype}")

#     choice = np.random.choice(current_points.shape[0], opt.npoints, replace=True)
#     seg = current_labels[choice]
#     print(f"Datatype of seg before conversion: {seg.dtype}")

#     point_set = current_points[choice, :]
#     seg = current_labels[choice]
#     seg = seg - seg.min()

#     gt = cmap[seg, :]
#     pred = cmap[seg, :]

#     while render_realtime(point_set, gt, c_pred=pred, freezerot=False, background=(255, 255, 255), normalizecolor=True, ballradius=opt.ballradius):
#         continue


def visualize_matplotlib(points, colors):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot with colors
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker='o')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud Visualization')
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../data/ss3dis/stanford_indoor3d', help='dataset path')
    parser.add_argument('--npoints', type=int, default=90000, help='resample points number')
    opt = parser.parse_args()

    cmap = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],  # Class 0
                    [3.12493437e-02, 1.00000000e+00, 1.31250131e-06],  # Class 1
                    [0.00000000e+00, 6.25019688e-02, 1.00000000e+00],  # Class 2
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 3
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 4
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 5
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 6
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 7
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 8
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 9
                    [0.50000000e+00, 0.50000000e+00, 0.00000000e+00],  # Class 10
                    [0.00000000e+00, 1.00000000e+00, 0.50000000e+00],  # Class 11
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]) # Class 12

    dataset = S3DISDataset(split='train', data_root='data/stanford_indoor3d', num_point=8192, test_area=5, block_size=1.0, sample_rate=1.0)

    idx = np.random.randint(0, len(dataset))
    current_points, current_labels = dataset[idx]

    # Convert the labels and select points for visualization
    current_labels = current_labels.astype(np.int32)
    choice = np.random.choice(current_points.shape[0], opt.npoints, replace=True)

    # Select the first 3 columns for (x, y, z) coordinates
    point_set = current_points[choice, :3]

    # Assign colors using cmap
    seg = current_labels[choice]
    gt_colors = cmap[seg, :]

    # Visualize using Matplotlib
    visualize_matplotlib(point_set, gt_colors)


# def visualize_open3d(points, colors):
#     # Ensure both points and colors are properly formatted
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     o3d.visualization.draw_geometries([pcd])

# if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../data/ss3dis/stanford_indoor3d', help='dataset path')
    parser.add_argument('--npoints', type=int, default=100, help='resample points number')
    opt = parser.parse_args()

    cmap = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],  # Class 0
                    [3.12493437e-02, 1.00000000e+00, 1.31250131e-06],  # Class 1
                    [0.00000000e+00, 6.25019688e-02, 1.00000000e+00],  # Class 2
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 3
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 4
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 5
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 6
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 7
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 8
                    [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],  # Class 9
                    [0.50000000e+00, 0.50000000e+00, 0.00000000e+00],  # Class 10
                    [0.00000000e+00, 1.00000000e+00, 0.50000000e+00],  # Class 11
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]) # Class 12

    dataset = S3DISDataset(split='train', data_root='data/stanford_indoor3d', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0)

    idx = np.random.randint(0, len(dataset))
    current_points, current_labels = dataset[idx]

    # Validate shapes before visualizing
    print(f"Shape of current_points: {current_points.shape}")
    print(f"Shape of current_labels: {current_labels.shape}")

    # Convert the labels and select points for visualization
    current_labels = current_labels.astype(np.int32)
    choice = np.random.choice(current_points.shape[0], opt.npoints, replace=True)

    # Select the first 3 columns for (x, y, z) coordinates
    point_set = current_points[choice, :3]

    # Assign colors using cmap
    seg = current_labels[choice]
    gt_colors = cmap[seg, :]

    # Ensure that both point_set and gt_colors have the correct data types
    point_set = point_set.astype(np.float64)  # Ensure points are float64
    gt_colors = gt_colors.astype(np.float32)  # Ensure colors are float32

    # Validate the color values and check for invalid values
    print(f"Minimum color value: {gt_colors.min()}")
    print(f"Maximum color value: {gt_colors.max()}")
    assert (gt_colors >= 0).all() and (gt_colors <= 1).all(), "Color values must be between 0 and 1"

    # Check for NaN or Inf values in point_set or gt_colors
    assert not np.isnan(point_set).any(), "NaN values found in point_set"
    assert not np.isnan(gt_colors).any(), "NaN values found in gt_colors"
    assert not np.isinf(point_set).any(), "Inf values found in point_set"
    assert not np.isinf(gt_colors).any(), "Inf values found in gt_colors"

    # Visualize using Open3D
    visualize_open3d(point_set, gt_colors)