import os
import h5py
import numpy as np
import open3d as o3d
from tqdm import tqdm

def save_pcd(xyzrgb_data, output_path):
    """
    Save the xyzrgb data as a .pcd file using Open3D.

    Args:
        xyzrgb_data (np.array): N x 6 array with xyzrgb data.
        output_path (str): Path to save the .pcd file.
    """
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    
    # Set points (xyz)
    pcd.points = o3d.utility.Vector3dVector(xyzrgb_data[:, :3])
    
    # Set colors (rgb), assuming the values are in range [0, 1]
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb_data[:, 3:6] / 255.0)
    
    # Save as .pcd
    o3d.io.write_point_cloud(output_path, pcd)

def process_h5_files(input_folder, output_folder, key='pointcloud'):
    """
    Process .h5 files, extract xyzrgb data, and save as .pcd files.

    Args:
        input_folder (str): Path to the folder containing input .h5 files.
        output_folder (str): Path to the folder to save the .pcd files.
        key (str): The key under which point cloud data is stored in .h5 files.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all .h5 files in the input folder
    h5_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
    
    # Iterate over the files with progress bar
    for h5_file in tqdm(h5_files, desc="Processing .h5 files", unit="file"):
        input_path = os.path.join(input_folder, h5_file)
        output_pcd_path = os.path.join(output_folder, h5_file.replace('.h5', '.pcd'))

        # Load the .h5 file and extract point cloud data
        with h5py.File(input_path, 'r') as f:
            data = f[key][:]  # xyzirgbl format
            
            # Extract xyzrgb (ignore intensity and labels)
            xyzrgb_data = data[:, [0, 1, 2, 4, 5, 6]]  # Extracting x, y, z, r, g, b

            # Save the extracted xyzrgb data as a .pcd file
            save_pcd(xyzrgb_data, output_pcd_path)

    print(f"Processing complete. Files saved to {output_folder}")

# Example usage
input_folder  = r'E:\pointnet2_pytorch_semantic\data\s3dis\buildings_h5_4_labels_z_rotated_y_mod_downsamp_0.2'  # Folder containing the original .h5 files
output_folder = r'E:\pointnet2_pytorch_semantic\data\s3dis\buildings_h5_4_labels_z_rotated_y_mod_downsamp_0.2_pcd'  # Folder to save the .pcd files

process_h5_files(input_folder, output_folder)
