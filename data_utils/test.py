import h5py
import numpy as np
# Replace 'your_file.h5' with the path to your .h5 file
file_path = '/mnt/e/pointnet2_pytorch_semantic/data/s3dis/buildings_h5_labels_fixed/B1505_NE_2all_processed_data.h5'
# Convert the dataset to float32 and save it
converted_data = {}
# Open the .h5 file
with h5py.File(file_path, 'r') as h5_file:
    # List all groups
    print("Keys in the file:", list(h5_file.keys()))

    # Access a specific dataset (replace 'dataset_name' with the actual dataset key)
    dataset = h5_file['pointcloud']
    # Convert the dataset to float32
    converted_data['pointcloud'] = np.array(dataset, dtype=np.float32)  

# Display the shape and new data type of the converted dataset
print(converted_data['pointcloud'].shape)
print(converted_data['pointcloud'].dtype)

new_file_path = '/mnt/e/pointnet2_pytorch_semantic/data/s3dis/buildings_h5_labels/B1505_NE_2all_processed_data.h5'
# Convert the 'block0_values' dataset to float32 and store it
converted_new_data = {}

with h5py.File(new_file_path, 'r') as h5_file:
        # List all groups
    print("Keys in the file:", list(h5_file.keys()))
    group = h5_file['processed_data']
    if 'block0_values' in group:
        dataset = group['block0_values']
        # Convert the dataset to float32
        converted_new_data['block0_values'] = np.array(dataset, dtype=np.float32)

# Display the shape and new data type of the converted dataset
print(converted_new_data['block0_values'].shape)
print(converted_new_data['block0_values'].dtype)
