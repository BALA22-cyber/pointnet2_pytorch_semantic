import numpy as np
import ctypes as ct
import sys
import os
import cv2
import argparse

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

# def render_to_image(xyz, c0, c1, c2, ballradius=10, background=(0, 0, 0), freezerot=False):
#     global showsz, zoom
#     show = np.zeros((showsz, showsz, 3), dtype='uint8')
    
#     rotmat = np.eye(3)
#     if not freezerot:
#         xangle = (mousey - 0.5) * np.pi * 1.2
#         yangle = (mousex - 0.5) * np.pi * 1.2
#     else:
#         xangle, yangle = 0, 0

#     rotmat = rotmat.dot(np.array([
#         [1.0, 0.0, 0.0],
#         [0.0, np.cos(xangle), -np.sin(xangle)],
#         [0.0, np.sin(xangle), np.cos(xangle)],
#     ]))
#     rotmat = rotmat.dot(np.array([
#         [np.cos(yangle), 0.0, -np.sin(yangle)],
#         [0.0, 1.0, 0.0],
#         [np.sin(yangle), 0.0, np.cos(yangle)],
#     ]))
#     rotmat *= zoom
#     nxyz = xyz.dot(rotmat) + [showsz / 2, showsz / 2, 0]

#     ixyz = nxyz.astype('int32')
#     show[:] = background
#     dll.render_ball(
#         ct.c_int(show.shape[0]),
#         ct.c_int(show.shape[1]),
#         show.ctypes.data_as(ct.c_void_p),
#         ct.c_int(ixyz.shape[0]),
#         ixyz.ctypes.data_as(ct.c_void_p),
#         c0.ctypes.data_as(ct.c_void_p),
#         c1.ctypes.data_as(ct.c_void_p),
#         c2.ctypes.data_as(ct.c_void_p),
#         ct.c_int(ballradius)
#     )
#     return show

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

def savepoints(xyz, c_gt=None, c_pred=None, filename='output.png', showrot=False, magnifyBlue=0,
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
    
    # Save the result
    cv2.imwrite(filename, img)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../data/ss3dis/stanford_indoor3d', help='dataset path')
    parser.add_argument('--category', type=str, default='door', help='select category')
    parser.add_argument('--npoints', type=int, default=2500, help='resample points number')
    parser.add_argument('--ballradius', type=int, default=1, help='ballradius')
    parser.add_argument('--output', type=str, default='output22.png', help='output image filename')
    opt = parser.parse_args()

    # cmap = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    #                  [3.12493437e-02, 1.00000000e+00, 1.31250131e-06],
    #                  [0.00000000e+00, 6.25019688e-02, 1.00000000e+00],
    #                  [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
    #                  [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
    #                  [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
    #                  [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
    #                  [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
    #                  [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
    #                  [1.00000000e+00, 0.00000000e+00, 9.37500000e-02]])

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
    # Ensure dataset is loaded correctly
    print(f"Total samples in dataset: {len(dataset)}")
    # Randomly select a point cloud from the dataset
    idx = np.random.randint(0, len(dataset))
    current_points, current_labels = dataset[idx]
    # Do something with current_points and current_labels
    print(f"Selected sample index: {idx}")
    print(f"Current points shape: {current_points.shape}")
    print(f"Current labels shape: {current_labels.shape}")
    # Check the datatype of the original current_labels
    print(f"Datatype of current_labels: {current_labels.dtype}")
    # Convert current_labels to integers
    current_labels = current_labels.astype(np.int32)
    print(f"Datatype of current_labels after conversion: {current_labels.dtype}")

    # Now use current_points instead of point_set and current_labels instead of seg
    choice = np.random.choice(current_points.shape[0], opt.npoints, replace=True)
    # After applying the choice selection, check the datatype of seg
    seg = current_labels[choice]
    print(f"Datatype of seg before conversion: {seg.dtype}")

    # # Convert to integer if necessary
    # seg = seg.astype(np.int32)
    # print(f"Datatype of seg after conversion: {seg.dtype}")


    point_set = current_points[choice, :]  # Use current_points instead of point_set
    seg = current_labels[choice]  # Use current_labels instead of seg
    seg = seg - seg.min()

    # Assign colors using the cmap for both ground truth and prediction
    gt = cmap[seg, :]
    pred = cmap[seg, :]

    savepoints(point_set, gt, c_pred=pred, filename=opt.output, showrot=False, magnifyBlue=0, freezerot=False,
               background=(255, 255, 255), normalizecolor=True, ballradius=opt.ballradius)
