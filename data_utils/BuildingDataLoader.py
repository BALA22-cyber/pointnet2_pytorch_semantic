import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, file_path, num_point=4096, transform=None):
        super().__init__()
        self.num_point = num_point
        self.transform = transform
        self.points, self.labels = self.load_h5_data(file_path)

    # def load_h5_data(self, file_path):
    #     # Open the file using Pandas to handle the DataFrame structure
    #     with pd.HDFStore(file_path) as store:
    #         # The 'processed_data' is the key where the DataFrame is stored
    #         data_df = store['processed_data']
            
    #     # Assuming that the DataFrame columns are well labeled according to your previous description
    #     points = data_df[['x', 'y', 'z', 'r', 'g', 'b']].values  # Convert desired columns to NumPy array
    #     labels = data_df['category'].values  # Assuming 'category' is the label column
        
    #     return points, labels.astype(np.int32)  # Ensure labels are of integer type

    # def load_h5_data(self, file_path):
    #     with h5py.File(file_path, 'r') as h5_file:
    #         # Access the group 'processed_data' and then the dataset directly
    #         # It seems the actual data might be stored in blocks
    #         # Let's assume 'block0_values' is what we need. You might need to adjust this
    #         if 'block0_values' in h5_file['processed_data']:
    #             data_array = h5_file['processed_data/block0_values'][:]
    #         else:
    #             raise ValueError("Dataset 'block0_values' not found in 'processed_data'")

    #         # Assuming the last column in the data array is labels
    #         points = data_array[:, :-1]  # All columns except the last one
    #         labels = data_array[:, -1]  # The last column

    #     return points, labels.astype(np.int32)
    
    def load_h5_data(self, file_path):
        with h5py.File(file_path, 'r') as h5_file:
            # Access the dataset, assuming 'block0_values' contains your data
            dataset = h5_file['processed_data/block0_values']
            
            # Convert dataset to a NumPy array with a more typical dtype if necessary
            data_array = np.array(dataset, dtype=np.float32)  # Using float32 for simplicity

            # Assuming the last column in the data array is labels
            points = data_array[:, :-1]  # All columns except the last one
            labels = data_array[:, -1]  # The last column

        return points, labels.astype(np.int32)

    def __getitem__(self, idx):
        # Shuffle points
        current_points = self.points[idx, :self.num_point]
        current_labels = self.labels[idx, :self.num_point]
        idxs = np.arange(current_points.shape[0])
        np.random.shuffle(idxs)

        current_points = current_points[idxs, :]
        current_labels = current_labels[idxs]

        if self.transform:
            current_points, current_labels = self.transform(current_points, current_labels)

        return current_points, current_labels

    def __len__(self):
        return self.points.shape[0]

if __name__ == '__main__':
    # Example usage
    dataset = CustomDataset(file_path='data/s3dis/buildings_h5_labels/B1505_NE_2all_processed_data.h5', num_point=4096, transform=None)
    print('Dataset size:', len(dataset))
    points, labels = dataset[0]  # Get the first sample
    print('First sample points shape:', points.shape)
    print('First sample labels shape:', labels.shape)


