import os
import h5py
import numpy as np
from tqdm import tqdm

def convert_h5_to_npy(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of all .h5 files in the input folder
    h5_files = [f for f in os.listdir(input_folder) if f.endswith(".h5")]

    # Loop over all .h5 files with a progress bar
    for filename in tqdm(h5_files, desc="Converting pointcloud to npy"):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename.replace('.h5', '.npy'))
        
        with h5py.File(input_file_path, 'r') as input_file:
            # Load the 'pointcloud' dataset
            pointcloud_data = np.array(input_file['pointcloud'], dtype=np.float32)
            
            # Save the data to an .npy file
            np.save(output_file_path, pointcloud_data)

# Usage
input_folder = '/mnt/e/pointnet2_pytorch_semantic/data/s3dis/buildings_h5_labels_fixed/'  # Folder containing the .h5 files
output_folder = '/mnt/e/pointnet2_pytorch_semantic/data/s3dis/buildings_h5_labels_npy/' # Folder where .npy files will be saved

convert_h5_to_npy(input_folder, output_folder)
