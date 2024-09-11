import h5py
import os
import numpy as np
import torch,time,random
from tqdm import tqdm
from torch.utils.data import Dataset

# class BuildingDataset(Dataset):
#     def __init__(self, data_root, num_point=4096, block_size=1.0, sample_rate=1.0, transform=None):
#         super().__init__()
#         self.num_point = num_point
#         self.block_size = block_size
#         self.transform = transform
#         self.files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.h5')]
        
#         self.facade_points, self.facade_labels = [], []
#         self.facade_coord_min, self.facade_coord_max = [], []
#         labelweights = np.zeros(3)  # Assuming 13 classes; adjust as needed
#         num_point_all = []

#         for file_path in self.files:
#             with h5py.File(file_path, 'r') as file:
#                 points = file['xyz'][:]  # xyz coordinates
#                 rgb = file['rgb'][:]  # RGB colors
#                 labels = file['l'][:]  # labels
#                 points = np.concatenate([points, rgb], axis=-1)  # Concatenate xyz with RGB

#                 tmp, _ = np.histogram(labels, range(4))
#                 labelweights += tmp
#                 coord_min = np.amin(points, axis=0)[:3]
#                 coord_max = np.amax(points, axis=0)[:3]

#                 self.facade_points.append(points)
#                 self.facade_labels.append(labels)
#                 self.facade_coord_min.append(coord_min)
#                 self.facade_coord_max.append(coord_max)
#                 num_point_all.append(labels.size)

#         labelweights = labelweights.astype(np.float32) / np.sum(labelweights)
#         self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        
#         sample_prob = num_point_all / np.sum(num_point_all)
#         num_iter = int(np.sum(num_point_all) * sample_rate / self.num_point)
#         facade_idxs = []
#         for i in range(len(self.files)):
#             facade_idxs += [i] * int(round(sample_prob[i] * num_iter))
#         self.facade_idxs = np.array(facade_idxs)

#     def __getitem__(self, idx):
#         facade_idx = self.facade_idxs[idx]
#         points = self.facade_points[facade_idx]   # N * 6 (xyzrgb)
#         labels = self.facade_labels[facade_idx]   # N
#         N_points = points.shape[0]

#         while True:
#             center = points[np.random.choice(N_points)][:3]
#             block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
#             block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
#             point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & 
#                                   (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
#             if point_idxs.size > 1024:
#                 break

#         selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=point_idxs.size < self.num_point)
#         selected_points = points[selected_point_idxs, :]
#         current_points = np.zeros((self.num_point, 9))
#         current_points[:, 6] = selected_points[:, 0] / self.facade_coord_max[facade_idx][0]
#         current_points[:, 7] = selected_points[:, 1] / self.facade_coord_max[facade_idx][1]
#         current_points[:, 8] = selected_points[:, 2] / self.facade_coord_max[facade_idx][2]
#         selected_points[:, 0] = selected_points[:, 0] - center[0]
#         selected_points[:, 1] = selected_points[:, 1] - center[1]
#         selected_points[:, 3:6] /= 255.0
#         current_points[:, 0:6] = selected_points
#         current_labels = labels[selected_point_idxs]
#         if self.transform:
#             current_points, current_labels = self.transform(current_points, current_labels)
#         return current_points, current_labels

#     def __len__(self):
#         return len(self.facade_idxs)

class BuildingDataset(Dataset):
    def __init__(self, data_root, split='train', num_point=4096, block_size=1.0,stride=0.5, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.stride = stride
        all_files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.h5')]
        
        # Split the data into training and testing datasets (80/20 split)
        # random.shuffle(all_files)
        split_index = int(len(all_files) * 0.8)
        self.files = all_files[:split_index] if split == 'train' else all_files[split_index:]
        
        self.facade_points, self.facade_labels = [], []
        self.facade_coord_min, self.facade_coord_max = [], []
        labelweights = np.zeros(3)  # Assuming 3 classes; adjust as needed
        num_point_all = []

        for file_path in tqdm(self.files, desc="Loading .h5 files"):
            with h5py.File(file_path, 'r') as file:
                data = file['pointcloud'][:]  # 'pointcloud' key contains xyzirgbl
                xyz = data[:, 0:3]  # XYZ coordinates
                intensity = data[:, 3:4]  # Intensity values
                rgb = data[:, 4:7]  # RGB colors
                labels = data[:, 7]  # Labels
                # points = np.concatenate([xyz, intensity, rgb], axis=-1) # include intensity?
                points = np.concatenate([xyz,rgb],axis = -1)

                tmp, _ = np.histogram(labels, range(4))
                labelweights += tmp
                coord_min = np.amin(points, axis=0)[:3]
                coord_max = np.amax(points, axis=0)[:3]

                self.facade_points.append(points)
                self.facade_labels.append(labels)
                self.facade_coord_min.append(coord_min)
                self.facade_coord_max.append(coord_max)
                num_point_all.append(len(labels))

        labelweights = labelweights.astype(np.float32) / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / self.num_point)
        facade_idxs = []
        for i in range(len(self.files)):
            facade_idxs += [i] * int(round(sample_prob[i] * num_iter))
        self.facade_idxs = np.array(facade_idxs)

    # def __getitem__(self, idx):
    """This method uses xyzrgbandnormals with random choice for center for processing blocks"""
    #     facade_idx = self.facade_idxs[idx]
    #     points = self.facade_points[facade_idx]
    #     labels = self.facade_labels[facade_idx]
    #     N_points = points.shape[0]

    #     while True:
    #         center = points[np.random.choice(N_points)][:3]
    #         block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
    #         block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
    #         point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) &
    #                               (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
    #         if point_idxs.size > 1024:
    #             break

    #     selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=point_idxs.size < self.num_point)
    #     selected_points = points[selected_point_idxs]
    #     # Adjusted to include all data columns properly
    #     current_points = np.zeros((self.num_point, 9))  # Now we use 9 to include xyz, intensity, rgb, and normalized xyz
    #     current_points[:, 6] = selected_points[:, 0] / self.facade_coord_max[facade_idx][0]
    #     current_points[:, 7] = selected_points[:, 1] / self.facade_coord_max[facade_idx][1]
    #     current_points[:, 8] = selected_points[:, 2] / self.facade_coord_max[facade_idx][2]
    #     selected_points[:, 0] = selected_points[:, 0] - center[0]
    #     selected_points[:, 1] = selected_points[:, 1] - center[1]
    #     selected_points[:, 3:6] /= 255.0  # Normalize RGB

    #     # 6 or 7? can intensity be used?
    #     current_points[:, 0:6] = selected_points

    #     current_labels = labels[selected_point_idxs]
    #     if self.transform:
    #         current_points, current_labels = self.transform(current_points, current_labels)
    #     return current_points, current_labels

    # def __getitem__(self, idx):
    #     """This method uses xyzrgb with random choice for center for processing blocks"""

    #     facade_idx = self.facade_idxs[idx]
    #     points = self.facade_points[facade_idx]
    #     labels = self.facade_labels[facade_idx]
    #     N_points = points.shape[0]

    #     while True:
    #         center = points[np.random.choice(N_points)][:3]
    #         block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
    #         block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
    #         point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) &
    #                             (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
    #         if point_idxs.size > 1024:
    #             break

    #     selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=point_idxs.size < self.num_point)
    #     selected_points = points[selected_point_idxs]
    #     current_points = np.zeros((self.num_point, 6))  # Adjust to 7 to include intensity with xyz and RGB

    #     # Perform the adjustments to the points
    #     selected_points[:, 0] = selected_points[:, 0] - center[0]  # Adjust X coordinates
    #     selected_points[:, 1] = selected_points[:, 1] - center[1]  # Adjust Y coordinates
    #     selected_points[:, 3:6] /= 255.0  # Normalize RGB values

    #     # Copy adjusted points into current_points
    #     current_points[:, 0:6] = selected_points

    #     current_labels = labels[selected_point_idxs]
    #     if self.transform:
    #         current_points, current_labels = self.transform(current_points, current_labels)
    #     return current_points, current_labels

    def __getitem__(self, idx):
        if idx >= len(self.facade_points) or idx >= len(self.facade_labels):
            raise ValueError(f"Index {idx} out of range. Max index: {len(self.facade_points)-1}")
        points = self.facade_points[idx]
        labels = self.facade_labels[idx]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        
        # Calculate how many blocks to divide the data into along each axis
        grid_x = int(np.ceil((coord_max[0] - coord_min[0] - self.block_size) / self.stride)) + 1
        grid_y = int(np.ceil((coord_max[1] - coord_min[1] - self.block_size) / self.stride)) + 1
        
        # Initialize containers to store batched data
        current_points, current_labels = [], []

        # Loop over each block in the grid
        for i in range(grid_x):
            for j in range(grid_y):
                s_x = coord_min[0] + i * self.stride
                s_y = coord_min[1] + j * self.stride
                
                # Define the bounds of the block
                block_min = np.array([s_x, s_y, 0])  # Assuming Z-min is constant
                block_max = block_min + self.block_size
                
                # Select points within this block
                selector = (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & \
                        (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1])
                if np.any(selector):
                    block_points = points[selector]
                    block_labels = labels[selector]
                    
                    # Normalize the points' coordinates within the block
                    block_points[:, 0:3] -= block_min[0:3]
                    block_points[:, 3:6] /= 255.0  # Normalize RGB

                    # Store the data for this block
                    current_points.append(block_points)
                    current_labels.append(block_labels)

        # Convert lists to arrays for training
        current_points = np.array(current_points, dtype=object)
        current_labels = np.array(current_labels, dtype=object)
        
        return current_points, current_labels


    # def __len__(self):
    #     return len(self.facade_idxs)
    def __len__(self):
        print(f"Dataset size: {len(self.facade_idxs)}")  # or appropriate length calculation
        return len(self.facade_idxs)


class ScannetDatasetWholeScene(Dataset):
    def __init__(self, root, num_point=4096, block_size=1.0, stride=0.5, padding=0.001):
        self.num_point = num_point
        self.block_size = block_size
        self.stride = stride
        self.padding = padding
        self.files = [os.path.join(root, file) for file in os.listdir(root) if file.endswith('.h5')]

        self.scene_points_list = []
        self.semantic_labels_list = []
        self.facade_coord_min = []
        self.facade_coord_max = []

        for file_path in self.files:
            with h5py.File(file_path, 'r') as file:
                points = file['xyz'][:]  # xyz coordinates
                rgb = file['rgb'][:]     # RGB values
                intensity = file['i'][:]  # Intensity values
                labels = file['l'][:]     # Labels

                points = np.concatenate([points, rgb], axis=-1)  # Concatenate xyz with RGB
                coord_min = np.amin(points, axis=0)[:3] - self.padding
                coord_max = np.amax(points, axis=0)[:3] + self.padding

                self.scene_points_list.append(np.concatenate([points, intensity[:, None]], axis=1))
                self.semantic_labels_list.append(labels)
                self.facade_coord_min.append(coord_min)
                self.facade_coord_max.append(coord_max)

        self.labelweights = self.calculate_labelweights()

    def __getitem__(self, index):
        points = self.scene_points_list[index]
        labels = self.semantic_labels_list[index]
        coord_min = self.facade_coord_min[index]
        coord_max = self.facade_coord_max[index]

        data_facade, label_facade, sample_weight, index_facade = [], [], [], []

        grid_x = int(np.ceil((coord_max[0] - coord_min[0] - self.block_size) / self.stride))
        grid_y = int(np.ceil((coord_max[1] - coord_min[1] - self.block_size) / self.stride))

        for index_x in range(grid_x + 1):
            for index_y in range(grid_y + 1):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])

                point_idxs = np.where((points[:, 0] >= s_x) & (points[:, 0] <= e_x) & 
                                      (points[:, 1] >= s_y) & (points[:, 1] <= e_y))[0]

                if point_idxs.size > 0:
                    selected_points = points[point_idxs]
                    selected_labels = labels[point_idxs]
                    weight = self.labelweights[selected_labels]

                    data_facade.append(selected_points)
                    label_facade.append(selected_labels)
                    sample_weight.append(weight)
                    index_facade.append(point_idxs)

        return np.concatenate(data_facade), np.concatenate(label_facade), np.concatenate(sample_weight), np.concatenate(index_facade)

    def calculate_labelweights(self):
        labelcounts = np.zeros(3)  # Assuming 3 classes
        for labels in self.semantic_labels_list:
            tmp, _ = np.histogram(labels, bins=np.arange(4))
            labelcounts += tmp
        labelweights = np.power(np.sum(labelcounts) / labelcounts, 1/3)
        labelweights[np.isinf(labelweights)] = 0
        return labelweights

    def __len__(self):
        return len(self.scene_points_list)



if __name__ == '__main__':
    # Define the path to your dataset
    data_root = '/mnt/e/pointnet2_pytorch_semantic/data/s3dis/buildings_3labels_downsamp_0.2'
    num_point, block_size, sample_rate,stride = 8192, 1.0,0.01 ,0.1

    point_data = BuildingDataset(split='train', data_root=data_root, num_point=num_point,stride=stride, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    print("before loading")
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=32, shuffle=True, num_workers=32, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()

    for i in range(len(data_root)):
        try:
            data, labels = data_root[i]
        except Exception as e:
            print(f"Failed to load data at index {i}: {str(e)}")