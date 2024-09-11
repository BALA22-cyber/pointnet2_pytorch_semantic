import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def get_class_distribution(folder_path, key='pointcloud'):
    class_counts = Counter()
    
    # List all the .h5 files in the folder
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    
    for h5_file in h5_files:
        file_path = os.path.join(folder_path, h5_file)
        
        with h5py.File(file_path, 'r') as h5f:
            # Read the point cloud data
            pointcloud = h5f[key][:]
            
            # Get the label column (assuming it's the last column in pointcloud)
            labels = pointcloud[:, -1]
            
            # Count the occurrences of each class
            class_counts.update(labels)
    
    return class_counts

def plot_class_distribution(class_counts):
    # Extract classes and counts
    classes = sorted(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
      # Calculate the total number of points
    total_points = sum(counts)
    
    # Calculate percentages for each class
    percentages = [(count / total_points) * 100 for count in counts]
    
    # Print the percentage distribution
    print("Class Distribution (Class: Count, Percentage):")
    for cls, count, percentage in zip(classes, counts, percentages):
        print(f"Class {int(cls)}: {count} points, {percentage:.2f}%")
    
    # Plot histogram
    plt.bar(classes, counts, color='red')
    plt.xlabel('Class Labels')
    plt.ylabel('Number of Points')
    plt.title('Class Distribution Across All .h5 Files')
    plt.xticks(classes)  # Ensure only class labels are shown on x-axis
    plt.show()

# Folder containing the .h5 files
# folder_path = r'E:\pointnet2_pytorch_semantic\data\s3dis\buildings_h5_4_labels'
folder_path = r'E:\pointnet2_pytorch_semantic\data\s3dis\buildings_2_labels_downsamp_0.2'

# Get class distribution
class_counts = get_class_distribution(folder_path)

# Plot class distribution
plot_class_distribution(class_counts)

"""for analyzing .npy files"""
# def get_class_distribution_from_npy(folder_path):
#     class_counts = Counter()
    
#     # List all the .npy files in the folder
#     npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    
#     for npy_file in npy_files:
#         file_path = os.path.join(folder_path, npy_file)
        
#         # Load the .npy file
#         data = np.load(file_path)
        
#         # Get the label column (assuming it's the last column in the array)
#         labels = data[:, -1]
        
#         # Count the occurrences of each class
#         class_counts.update(labels)
    
#     return class_counts

# def plot_class_distribution(class_counts):
#     # Extract classes and counts
#     classes = sorted(class_counts.keys())
#     counts = [class_counts[cls] for cls in classes]
    
#     # Calculate the total number of points
#     total_points = sum(counts)
    
#     # Calculate percentages for each class
#     percentages = [(count / total_points) * 100 for count in counts]
    
#     # Print the percentage distribution
#     print("Class Distribution (Class: Count, Percentage):")
#     for cls, count, percentage in zip(classes, counts, percentages):
#         print(f"Class {int(cls)}: {count} points, {percentage:.2f}%")
    
#     # Plot histogram
#     plt.bar(classes, counts, color='green')
#     plt.xlabel('Class Labels')
#     plt.ylabel('Number of Points')
#     plt.title('Class Distribution Across All .npy Files')
#     plt.xticks(classes)  # Ensure only class labels are shown on x-axis
#     plt.show()

# # Folder containing the .npy files
# folder_path = 'data/s3dis/buildings_h5_4_labels'

# # Get class distribution from .npy files
# class_counts = get_class_distribution_from_npy(folder_path)

# # Plot class distribution and print percentages
# plot_class_distribution(class_counts)