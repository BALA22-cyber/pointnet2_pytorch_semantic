import os
import h5py
import numpy as np
from tqdm import tqdm

def rotate_z_90(pointcloud):
    # Define a 90-degree rotation matrix around the z-axis
    rotation_matrix = np.array([[0, -1, 0], 
                                [1,  0, 0], 
                                [0,  0, 1]])

    # Apply the rotation matrix to the xyz coordinates (first three columns)
    pointcloud[:, :3] = np.dot(pointcloud[:, :3], rotation_matrix.T)

    return pointcloud

def process_h5_files_for_rotation(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all .h5 files in the folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
    files.sort()  # Ensure consistent file order
    # Use tqdm for a progress bar
    for file_name in tqdm(files, desc="Processing files"):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)

        with h5py.File(input_file_path, 'r') as h5_file:
            # Load the 'pointcloud' dataset
            pointcloud = h5_file['pointcloud'][:]

            # Rotate the point cloud around the z axis by 90 degrees
            pointcloud = rotate_z_90(pointcloud)

        # Save the rotated point cloud in the new folder
        with h5py.File(output_file_path, 'w') as new_h5_file:
            new_h5_file.create_dataset('pointcloud', data=pointcloud)

        print(f"Processed and rotated file {file_name} by 90 degrees and saved to {output_file_path}")

# Specify the input and output folder
input_folder = r'E:\pointnet2_pytorch_semantic\data\s3dis\buildings_h5_4_labels'
output_folder = r'E:\pointnet2_pytorch_semantic\data\s3dis\buildings_h5_4_labels_z_rotated'

process_h5_files_for_rotation(input_folder, output_folder)
