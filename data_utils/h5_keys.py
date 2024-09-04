# import h5py

# def list_keys_in_h5_file(file_path):
#     with h5py.File(file_path, 'r') as file:
#         print(list(file.keys()))

# if __name__ == "__main__":
#     file_path = 'data/s3dis/buildings_h5_labels/B1505_NE_2all_processed_data.h5'
#     list_keys_in_h5_file(file_path)


import h5py

# def inspect_h5_file(file_path):
#     with h5py.File(file_path, 'r') as h5_file:
#         print("Keys in the file:")
#         print(list(h5_file.keys()))
#         print("\nDetails of 'processed_data':")
#         try:
#             for item in h5_file['processed_data']:
#                 print(item)
#         except Exception as e:
#             print(e)

# if __name__ == "__main__":
#     file_path = 'data/s3dis/buildings_h5_labels/B1505_NE_2all_processed_data.h5'
#     inspect_h5_file(file_path)

import h5py
import numpy as np

def read_and_cast_data(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        # Access the dataset
        dataset = h5_file['processed_data/block0_values']

        # Attempt to cast data directly to a float32 numpy array to avoid precision issues
        try:
            data_array = dataset[:].astype(np.float32)
        except Exception as e:
            print("Error during casting:", e)
            return None, None

        # Assuming the last column is labels and should be integers
        points = data_array[:, :-1]  # Select all columns except the last
        labels = data_array[:, -1]   # Select the last column as labels
        labels = labels.astype(np.int32)  # Convert labels to integer type

        return points, labels

if __name__ == "__main__":
    file_path = 'data/s3dis/buildings_h5_labels/B1505_NE_2all_processed_data.h5'
    points, labels = read_and_cast_data(file_path)
    if points is not None and labels is not None:
        print('First point sample:', points[0])
        print('First label sample:', labels[0])



