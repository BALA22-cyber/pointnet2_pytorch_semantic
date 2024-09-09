import h5py
import numpy as np
import os

def load_and_modify_labels(h5_file_path, output_file_path=None):
    # Load the dataset
    with h5py.File(h5_file_path, 'r') as f:
        # Assuming 'pointcloud' contains all the data in xyzirgbl format
        pointcloud_data = f['pointcloud'][:]
        
        # Extract the labels (last column)
        labels = pointcloud_data[:, -1]   # Labels are in the last column
        
        # Modify the labels
        labels[labels == 2] = 0  # Rewrite label 2 as label 0
        labels[labels == 3] = 2  # Rewrite label 3 as label 2
        
        # Update the labels back in the pointcloud_data
        pointcloud_data[:, -1] = labels
    
    # Optionally save the modified data to a new file
    if output_file_path:
        with h5py.File(output_file_path, 'w') as f:
            f.create_dataset('pointcloud', data=pointcloud_data)
        print(f"Modified dataset saved to {output_file_path}")
    
    return pointcloud_data

def process_all_h5_files(input_folder, output_folder=None):
    # List all .h5 files in the input folder
    h5_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
    
    if not h5_files:
        print(f"No .h5 files found in {input_folder}")
        return
    
    # Ensure output folder exists
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each file
    for h5_file in h5_files:
        input_file_path = os.path.join(input_folder, h5_file)
        
        # Create output path for each file if output_folder is specified
        if output_folder:
            output_file_path = os.path.join(output_folder, h5_file)
        else:
            output_file_path = None  # Skip saving if output folder isn't specified
        
        print(f"Processing {h5_file}...")
        load_and_modify_labels(input_file_path, output_file_path)

# Usage
input_folder = r'E:\pointnet2_pytorch_semantic\data\s3dis\buildings_downsamp_0.2'  # Replace with the path to the folder containing .h5 files
output_folder = r'E:\pointnet2_pytorch_semantic\data\s3dis\buildings_3label_downsamp_0.2'  # Replace with the path to the output folder, or None to skip saving
process_all_h5_files(input_folder, output_folder)
