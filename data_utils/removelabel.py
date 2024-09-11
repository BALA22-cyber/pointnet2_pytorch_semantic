import os
import h5py
import numpy as np

def remove_class_3_from_h5_files(folder_path, key='pointcloud'):
    # List all the .h5 files in the folder
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    
    for h5_file in h5_files:
        file_path = os.path.join(folder_path, h5_file)
        
        with h5py.File(file_path, 'r+') as h5f:
            # Read the point cloud data
            pointcloud = h5f[key][:]
            
            # Get the label column (assuming it's the last column in pointcloud)
            labels = pointcloud[:, -1]
            
            # Keep rows where the label is not equal to 3
            filtered_pointcloud = pointcloud[labels != 3]
            
            # Delete the existing dataset
            del h5f[key]
            
            # Write the filtered pointcloud back to the same key
            h5f.create_dataset(key, data=filtered_pointcloud)
        
        print(f"Processed file: {h5_file}")

# # Folder containing the .h5 files
# folder_path = '/mnt/e/pointnet2_pytorch_semantic/data/s3dis/buildings_h5_4_labels/'

# # Call the function
# #remove_class_3_from_h5_files(folder_path)

def convert_label_4_to_3(folder_path, key='pointcloud'):
    # List all the .h5 files in the folder
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    
    for h5_file in h5_files:
        file_path = os.path.join(folder_path, h5_file)
        
        with h5py.File(file_path, 'r+') as h5f:
            # Read the point cloud data
            pointcloud = h5f[key][:]
            
            # Get the label column (assuming it's the last column in pointcloud)
            labels = pointcloud[:, -1]
            
            # Convert all label 4 to label 3
            labels[labels == 4] = 3
            
            # Update the pointcloud with the modified labels
            pointcloud[:, -1] = labels
            
            # Delete the existing dataset and write the updated pointcloud back
            del h5f[key]
            h5f.create_dataset(key, data=pointcloud)
        
        print(f"Processed file: {h5_file}")

# Folder containing the .h5 files
folder_path = 'data/s3dis/buildings_h5_4_labels/'
E:\pointnet2_pytorch_semantic\data\s3dis\buildings_h5_labels

# Call the function to convert label 4 to label 3
convert_label_4_to_3(folder_path)
