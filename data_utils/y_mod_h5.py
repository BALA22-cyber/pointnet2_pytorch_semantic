import os
import h5py
import numpy as np

def process_h5_files(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all .h5 files in the folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
    files.sort()  # Ensure consistent file order

    for idx, file_name in enumerate(files):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)

        with h5py.File(input_file_path, 'r') as h5_file:
            # Load the 'pointcloud' dataset
            pointcloud = h5_file['pointcloud'][:]
            
            # Increment y values by 5*(idx+1), y is the second column (index 1)
            pointcloud[:, 1] += 5 * (idx + 1)
        
        # Save the modified pointcloud in the new folder
        with h5py.File(output_file_path, 'w') as new_h5_file:
            new_h5_file.create_dataset('pointcloud', data=pointcloud)

        print(f"Processed file {file_name} with y offset {5 * (idx + 1)} and saved to {output_file_path}")

# Specify the input and output folder
input_folder = r'E:\pointnet2_pytorch_semantic\data\s3dis\buildings_h5_4_labels_z_rotated'
output_folder= r'E:\pointnet2_pytorch_semantic\data\s3dis\buildings_h5_4_labels_z_rotated_y_modified'

process_h5_files(input_folder, output_folder)
