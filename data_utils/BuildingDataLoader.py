import os
import numpy as np
import h5py
import torch,time,random
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

class BuildingDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.split = split
        self.file_list = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.h5')]
        self.data, self.labels = self.load_data()
        
                # Perform the train-test split based on the 'split' argument
        if self.split == 'train':
            self.train_data, self.train_labels = self.perform_train_split()
        else:
            self.test_data, self.test_labels = self.perform_test_split()
                # Calculate labelweights
        self.labelweights = self.calculate_labelweights()

    def load_data(self):
        all_data = []
        all_labels = []
        for h5_file in self.file_list:
            with h5py.File(h5_file, 'r') as f:
                # Explicitly cast to float32 to avoid precision issues
                dataset = np.array(f['pointcloud'], dtype=np.float32)

                # Extracting x, y, z, r, g, b (excluding intensity, which is index 3)
                data = dataset[:, [0, 1, 2, 4, 5, 6]]
                # print(f"Shape of data from {h5_file}: {data.shape}")  # Debug print
                
                # Extracting the labels (last column)
                labels = dataset[:, 7]
                # print(f"Shape of labels from {h5_file}: {labels.shape}")  # Debug print
                
                all_data.append(data)
                all_labels.append(labels)
        all_data = np.concatenate(all_data, axis=0)#.reshape(-1, 6)
        all_labels = np.concatenate(all_labels, axis=0)#.reshape(-1) 
        print(f"Final shape of all_data: {all_data.shape}")  # Debug print
        print(f"Final shape of all_labels: {all_labels.shape}")  # Debug print
        return all_data, all_labels
    
    def calculate_labelweights(self):
        # There are 5 classes: window (0), wall (1), door (2), vent (3), others (4)
        num_classes = 5
        label_histogram = np.zeros(num_classes)

        # Count the number of occurrences of each class in the training set
        if self.split == 'train':
            labels = self.train_labels
        else:
            labels = self.test_labels

        for label in labels:
            label_histogram[int(label)] += 1

        # Normalize the label frequencies
        total_labels = np.sum(label_histogram)
        label_histogram = label_histogram / total_labels

        # Avoid divide by zero by replacing zero values with a small number
        label_histogram[label_histogram == 0] = 1e-6  # Set a small value for missing classes

        # Calculate label weights: inversely proportional to the frequency, raised to 1/3 power
        labelweights = np.power(np.amax(label_histogram) / label_histogram, 1 / 3.0)

        print(f"Label weights: {labelweights}")
        return labelweights

    
    def perform_train_split(self):
        # Example logic for train split: Select 80% of the data for training
        total_points = len(self.data)
        split_idx = int(0.8 * total_points)
        train_data = self.data[:split_idx]
        train_labels = self.labels[:split_idx]
        return train_data, train_labels
    
    def perform_test_split(self):
        # Example logic for test split: Select the remaining 20% of the data for testing
        total_points = len(self.data)
        split_idx = int(0.8 * total_points)
        test_data = self.data[split_idx:]
        test_labels = self.labels[split_idx:]
        return test_data, test_labels
    
    def __getitem__(self, idx):

        if self.split == 'train':
            data = self.train_data
            labels = self.train_labels
        else:
            data = self.test_data
            labels = self.test_labels

        # Sample a block of points
        start_idx = idx * self.num_point
        end_idx = (idx + 1) * self.num_point

        points = self.data[start_idx:end_idx]  # Extract a block of points (num_point points)
        labels = self.labels[start_idx:end_idx]

        if points.shape[0] == 0:
            # If no points are found, raise an exception
            raise ValueError(f"No points found for index {idx}. Check dataset length and indexing.")
        
        if points.shape[0] < self.num_point:
            # If there are fewer than num_point points, oversample to make up the required number
            selected_point_idxs = np.random.choice(points.shape[0], self.num_point, replace=True)
            points = points[selected_point_idxs]
            labels = labels[selected_point_idxs]

        # Convert points and labels to PyTorch tensors using torch.as_tensor()
        points = torch.as_tensor(points, dtype=torch.float32)  
        labels = torch.as_tensor(labels, dtype=torch.float32)  
        
        # Optionally, apply transformation if any (e.g., augmentation)
        if self.transform is not None:
            points, labels = self.transform(points, labels)
            
        return points, labels

    def __len__(self):
        if self.split == 'train':
            return len(self.train_data) // self.num_point
        else:
            return len(self.test_data) // self.num_point

if __name__ == '__main__':

    data_root = '/mnt/e/pointnet2_pytorch_semantic/data/s3dis/buildings_h5_labels_fixed'  
    num_point,block_size,sample_rate = 4096, 1.0, 0.01

    # Initialize the dataset for training
    train_data = BuildingDataset(split='train', data_root=data_root, num_point=num_point, block_size=block_size, sample_rate=sample_rate, transform=None)
    
    print('Training data size:', len(train_data))
    print('Training point data 0 shape:', train_data.__getitem__(0)[0].shape)
    print('Training point label 0 shape:', train_data.__getitem__(0)[1].shape)

    # Initialize the dataset for testing
    test_data = BuildingDataset(split='test', data_root=data_root, num_point=num_point, block_size=block_size, sample_rate=sample_rate, transform=None)
    
    print('Testing data size:', len(test_data))
    print('Testing point data 0 shape:', test_data.__getitem__(0)[0].shape)
    print('Testing point label 0 shape:', test_data.__getitem__(0)[1].shape)

        # Set random seeds for reproducibility
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    
    
    # You can now use train_loader and test_loader with the respective datasets
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    # Iterate over the training dataset
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()


# if __name__ == '__main__':

#     data_root = '/mnt/e/pointnet2_pytorch_semantic/data/s3dis/buildings_h5_labels_fixed'  
#     num_point,block_size,sample_rate = 4096, 1.0, 0.01

#     # Initialize the dataset
#     point_data = BuildingDataset(data_root=data_root, num_point=num_point, block_size=block_size, sample_rate=sample_rate, transform=None)
    
#     print('point data size:', len(point_data))
#     print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
#     print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    
#     # Set random seeds for reproducibility
#     manual_seed = 123
#     random.seed(manual_seed)
#     np.random.seed(manual_seed)
#     torch.manual_seed(manual_seed)
#     torch.cuda.manual_seed_all(manual_seed)
    
#     def worker_init_fn(worker_id):
#         random.seed(manual_seed + worker_id)
    
#     # Initialize the DataLoader with your dataset
#     train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    
#     # Iterate over the dataset and measure time per batch
#     for idx in range(4):
#         end = time.time()
#         for i, (input, target) in enumerate(train_loader):
#             print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
#             end = time.time()

