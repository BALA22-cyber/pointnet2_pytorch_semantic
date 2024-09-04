""" Original Author: Haoqiang Fan """
import numpy as np
import ctypes as ct
import cv2
import sys
import os
import argparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
showsz = 800
mousex, mousey = 0.5, 0.5
zoom = 1.0
changed = True

def onmouse(*args):
    global mousex, mousey, changed
    y = args[1]
    x = args[2]
    mousex = x / float(showsz)
    mousey = y / float(showsz)
    changed = True


cv2.namedWindow('show3d')
cv2.moveWindow('show3d', 0, 0)
cv2.setMouseCallback('show3d', onmouse)

dll = np.ctypeslib.load_library(os.path.join(BASE_DIR, 'render_balls_so'), '.')


def showpoints(xyz, c_gt=None, c_pred=None, waittime=0, showrot=False, magnifyBlue=0, freezerot=False,
               background=(0, 0, 0), normalizecolor=True, ballradius=10):
    global showsz, mousex, mousey, zoom, changed
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

    show = np.zeros((showsz, showsz, 3), dtype='uint8')

    def render():
        rotmat = np.eye(3)
        if not freezerot:
            xangle = (mousey - 0.5) * np.pi * 1.2
        else:
            xangle = 0
        rotmat = rotmat.dot(np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(xangle), -np.sin(xangle)],
            [0.0, np.sin(xangle), np.cos(xangle)],
        ]))
        if not freezerot:
            yangle = (mousex - 0.5) * np.pi * 1.2
        else:
            yangle = 0
        rotmat = rotmat.dot(np.array([
            [np.cos(yangle), 0.0, -np.sin(yangle)],
            [0.0, 1.0, 0.0],
            [np.sin(yangle), 0.0, np.cos(yangle)],
        ]))
        rotmat *= zoom
        # nxyz = xyz.dot(rotmat) + [showsz / 2, showsz / 2, 0]
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

        if magnifyBlue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=0))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=0))
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=1))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=1))
        if showrot:
            cv2.putText(show, 'xangle %d' % (int(xangle / np.pi * 180)), (30, showsz - 30), 0, 0.5,
                        cv2.cv.CV_RGB(255, 0, 0))
            cv2.putText(show, 'yangle %d' % (int(yangle / np.pi * 180)), (30, showsz - 50), 0, 0.5,
                        cv2.cv.CV_RGB(255, 0, 0))
            cv2.putText(show, 'zoom %d%%' % (int(zoom * 100)), (30, showsz - 70), 0, 0.5, cv2.cv.CV_RGB(255, 0, 0))

    changed = True
    while True:
        if changed:
            render()
            changed = False
        cv2.imshow('show3d', show)
        if waittime == 0:
            cmd = cv2.waitKey(10) % 256
        else:
            cmd = cv2.waitKey(waittime) % 256
        if cmd == ord('q'):
            break
        elif cmd == ord('Q'):
            sys.exit(0)

        if cmd == ord('t') or cmd == ord('p'):
            if cmd == ord('t'):
                if c_gt is None:
                    c0 = np.zeros((len(xyz),), dtype='float32') + 255
                    c1 = np.zeros((len(xyz),), dtype='float32') + 255
                    c2 = np.zeros((len(xyz),), dtype='float32') + 255
                else:
                    c0 = c_gt[:, 0]
                    c1 = c_gt[:, 1]
                    c2 = c_gt[:, 2]
            else:
                if c_pred is None:
                    c0 = np.zeros((len(xyz),), dtype='float32') + 255
                    c1 = np.zeros((len(xyz),), dtype='float32') + 255
                    c2 = np.zeros((len(xyz),), dtype='float32') + 255
                else:
                    c0 = c_pred[:, 0]
                    c1 = c_pred[:, 1]
                    c2 = c_pred[:, 2]
            if normalizecolor:
                c0 /= (c0.max() + 1e-14) / 255.0
                c1 /= (c1.max() + 1e-14) / 255.0
                c2 /= (c2.max() + 1e-14) / 255.0
            c0 = np.require(c0, 'float32', 'C')
            c1 = np.require(c1, 'float32', 'C')
            c2 = np.require(c2, 'float32', 'C')
            changed = True

        if cmd == ord('n'):
            zoom *= 1.1
            changed = True
        elif cmd == ord('m'):
            zoom /= 1.1
            changed = True
        elif cmd == ord('r'):
            zoom = 1.0
            changed = True
        elif cmd == ord('s'):
            cv2.imwrite('show3d.png', show)
        if waittime != 0:
            break
    return cmd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../data/stanford_3d', help='dataset path')
    parser.add_argument('--category', type=str, default='window', help='select category')
    parser.add_argument('--npoints', type=int, default=250000, help='resample points number')
    parser.add_argument('--ballradius', type=int, default=1, help='ballradius')
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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    sys.path.append(BASE_DIR)
    sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))

    # from ShapeNetDataLoader import PartNormalDataset
    # root = '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    # dataset = PartNormalDataset(root = root, npoints=2048, split='test', normal_channel=False)
    # idx = np.random.randint(0, len(dataset))
    # data = dataset[idx]
    # point_set, _, seg = data
    # choice = np.random.choice(point_set.shape[0], opt.npoints, replace=True)
    # point_set, seg = point_set[choice, :], seg[choice]
    # seg = seg - seg.min()
    # gt = cmap[seg, :]
    # pred = cmap[seg, :]

    from S3DISDataLoader import S3DISDataset
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
    point_set = current_points[choice, :]  # Use current_points instead of point_set
    seg = current_labels[choice]  # Use current_labels instead of seg
    seg = seg - seg.min()
        # Assign colors using the cmap for both ground truth and prediction
    gt = cmap[seg, :]
    pred = cmap[seg, :]

    showpoints(point_set, gt, c_pred=pred, waittime=0, showrot=False, magnifyBlue=0, freezerot=False,
               background=(255, 255, 255), normalizecolor=True, ballradius=opt.ballradius)       


  


