import os
import numpy as np
import torch,time,random
from tqdm import tqdm
from torch.utils.data import Dataset

class CustomNPYDataset(Dataset):
    def __init__(self, data_root, num_point=4096, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.file_list = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.npy')]
        self.data, self.labels = self.load_data()

    def load_data(self):
        all_data = []
        all_labels = []
        for npy_file in self.file_list:
            # Load the data from the .npy file
            dataset = np.load(npy_file)

            # Assuming the format is the same: x, y, z, intensity, r, g, b, label
            # Extracting x, y, z, r, g, b (excluding intensity, index 3)
            data = dataset[:, [0, 1, 2, 4, 5, 6]]
            
            # Extracting the labels (assuming they are in the last column)
            labels = dataset[:, 7]
            
            all_data.append(data)
            all_labels.append(labels)

        # Concatenate all data and labels into a single array
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return all_data, all_labels

    def __getitem__(self, idx):
        points = self.data[idx][:, :6]  # Extract x, y, z, r, g, b
        labels = self.labels[idx]       # Extract label

        N_points = points.shape[0]
        if N_points >= self.num_point:
            selected_point_idxs = np.random.choice(N_points, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(N_points, self.num_point, replace=True)

        selected_points = points[selected_point_idxs, :]
        current_points = np.zeros((self.num_point, 6))  # num_point * 6 for x, y, z, r, g, b
        current_points[:, 0:6] = selected_points[:, 0:6]

        current_labels = labels[selected_point_idxs]

        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)

        return current_points, current_labels

    def __len__(self):
        return len(self.data)

# Example usage:
if __name__ == '__main__':
    data_root = '/mnt/e/pointnet2_pytorch_semantic/data/s3dis/buildings_h5_labels_fixed' #
    num_point = 4096
    block_size = 1.0
    sample_rate = 0.01

    # Initialize dataset
    point_data = CustomNPYDataset(data_root=data_root, num_point=num_point, block_size=block_size, sample_rate=sample_rate, transform=None)
    
    # Print dataset info
    print('point data size:', len(point_data))
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)

    # Example DataLoader setup
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)

    train_loader = torch.utils.data.DataLoader(point_data, batch_size=32, shuffle=True, num_workers=32, pin_memory=True, worker_init_fn=worker_init_fn)

    # Iterate over dataset
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()
