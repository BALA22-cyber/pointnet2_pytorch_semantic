import os
import h5py
import numpy as np
from tqdm import tqdm
from collections import Counter
import shutil

def calculate_class_distribution(labels, num_classes=4):
    """ Calculate the percentage distribution of classes. """
    class_counts = Counter(labels)
    total_labels = sum(class_counts.values())
    class_distribution = {i: (class_counts.get(i, 0) / total_labels) * 100 for i in range(num_classes)}
    return class_distribution

def downsample_class(points, labels, target_label, downsample_ratio):
    """
    Downsample the points of a specific class (target_label) by the given downsample_ratio.
    
    Args:
        points (np.array): Array of points (Nx7, where the last column is the label).
        labels (np.array): Array of corresponding labels (Nx1).
        target_label (int): The label for the class you want to downsample.
        downsample_ratio (float): The ratio to downsample the target class.
        
    Returns:
        downsampled_points (np.array): Points after downsampling.
        downsampled_labels (np.array): Corresponding labels after downsampling.
    """
    # Separate the points based on the target label (e.g., walls class 1)
    target_points = points[labels == target_label]
    target_labels = labels[labels == target_label]
    
    # Keep the points from the other classes unchanged
    non_target_points = points[labels != target_label]
    non_target_labels = labels[labels != target_label]
    
    # Calculate the number of points to retain from the target class
    num_target_points = target_points.shape[0]
    num_points_to_keep = int(num_target_points * downsample_ratio)
    
    # Randomly select the indices of points to keep from the target class
    selected_indices = np.random.choice(num_target_points, num_points_to_keep, replace=False)
    
    # Downsample the target points
    downsampled_target_points = target_points[selected_indices]
    downsampled_target_labels = target_labels[selected_indices]
    
    # Combine the downsampled target points with the non-target points
    downsampled_points = np.vstack((downsampled_target_points, non_target_points))
    downsampled_labels = np.hstack((downsampled_target_labels, non_target_labels))
    
    return downsampled_points, downsampled_labels

def process_h5_files(input_folder, output_folder, target_label, downsample_ratio, key='pointcloud'):
    """
    Process each .h5 file in the input folder, downsample points of the target label, and save the output.
    
    Args:
        input_folder (str): Path to the folder containing input .h5 files.
        output_folder (str): Path to the folder to save the processed .h5 files.
        target_label (int): The label for the class you want to downsample.
        downsample_ratio (float): The ratio to downsample the target class.
        key (str): The key under which point cloud data is stored in .h5 files.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all .h5 files in the input folder
    h5_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
    
    # Iterate over the files with progress bar
    for h5_file in tqdm(h5_files, desc="Processing files", unit="file"):
        input_path = os.path.join(input_folder, h5_file)
        output_path = os.path.join(output_folder, h5_file)

        with h5py.File(input_path, 'r') as f:
            data = f[key][:]
            points, labels = data[:, :-1], data[:, -1]  # Assume labels are in the last column

            # Calculate and print the class distribution before downsampling
            class_distribution_before = calculate_class_distribution(labels)
            print(f"Class distribution before downsampling for {h5_file}: {class_distribution_before}")

            # Downsample the target class (class 1)
            downsampled_points, downsampled_labels = downsample_class(points, labels, target_label, downsample_ratio)
            
            # Reconstruct the dataset by appending labels back
            downsampled_data = np.hstack((downsampled_points, downsampled_labels.reshape(-1, 1)))

            # Calculate and print the class distribution after downsampling
            class_distribution_after = calculate_class_distribution(downsampled_labels)
            print(f"Class distribution after downsampling for {h5_file}: {class_distribution_after}")
            
            # Save the downsampled point cloud back to .h5 format
            with h5py.File(output_path, 'w') as h5f:
                h5f.create_dataset(key, data=downsampled_data)

    print("Processing complete. Files saved to:", output_folder)


if __name__ == "__main__":
    input_folder = r'E:\pointnet2_pytorch_semantic\data\s3dis\buildings_h5_4_labels'
    output_folder = r'E:\pointnet2_pytorch_semantic\data\s3dis\buildings_downsamp_0.2'
    downsample_ratio = 0.2  # Downsample class 1 (walls) to 50% of its original size
    target_label = 1  # Class label for walls
    process_h5_files(input_folder, output_folder, target_label, downsample_ratio)


# if __name__ == "__main__":
#     input_folder = r'E:\pointnet2_pytorch_semantic\data\s3dis\buildings_h5_4_labels'
#     output_folder = r'E:\pointnet2_pytorch_semantic\data\s3dis\buildings_h5_wall_downsampled'
    
#     downsample_ratio = 0.5  # Reduce the "walls" category by 50%
#     process_and_downsample_h5_files(input_folder, output_folder, downsample_ratio=downsample_ratio, target_label=1)
