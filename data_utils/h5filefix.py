import os
import h5py
import numpy as np
from tqdm import tqdm

def convert_h5_files(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of all .h5 files in the input folder
    h5_files = [f for f in os.listdir(input_folder) if f.endswith(".h5")]

    # Loop over all .h5 files with a progress bar
    for filename in tqdm(h5_files, desc="Converting H5 Files"):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)
        
        with h5py.File(input_file_path, 'r') as input_file:
            with h5py.File(output_file_path, 'w') as output_file:
                # Copy the group and datasets structure
                group = input_file['processed_data']
                output_group = output_file.create_group('processed_data')
                
                # Copy axis0, axis1, block0_items safely (handle string datasets separately)
                for key in ['axis0', 'axis1', 'block0_items']:
                    dataset = group[key]
                    
                    # Check if the dataset has a valid shape
                    if dataset.size == 0:
                        print(f"Skipping empty dataset: {key}")
                        continue
                    
                    # Handle string datasets separately
                    if dataset.dtype.kind == 'S':  # Check if it's a string dataset
                        output_group.create_dataset(key, data=dataset[:].astype(str))
                    else:
                        output_group.create_dataset(key, data=dataset[:])
                
                # Convert block0_values dataset to float32 and save in new file
                block0_values = group['block0_values']
                converted_block0_values = np.array(block0_values, dtype=np.float32)
                output_group.create_dataset('block0_values', data=converted_block0_values)


def convert_block0_values(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of all .h5 files in the input folder
    h5_files = [f for f in os.listdir(input_folder) if f.endswith(".h5")]

    # Loop over all .h5 files with a progress bar
    for filename in tqdm(h5_files, desc="Converting block0_values to pointcloud"):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)
        
        with h5py.File(input_file_path, 'r') as input_file:
            group = input_file['processed_data']
            
            # Extract and convert block0_values dataset to float32
            block0_values = group['block0_values']
            converted_block0_values = np.array(block0_values, dtype=np.float32)
            
            # Create a new .h5 file and save the converted block0_values under the key 'pointcloud'
            with h5py.File(output_file_path, 'w') as output_file:
                output_file.create_dataset('pointcloud', data=converted_block0_values)

# Usage
input_folder = '/mnt/e/pointnet2_pytorch_semantic/data/s3dis/buildings_h5_labels/'  # Folder containing the original .h5 files
output_folder = '/mnt/e/pointnet2_pytorch_semantic/data/s3dis/buildings_h5_labels_fixed/'  # Folder where converted .h5 files will be saved
convert_block0_values(input_folder, output_folder)
# convert_h5_files(input_folder, output_folder)
