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
    
    # Plot histogram
    plt.bar(classes, counts, color='blue')
    plt.xlabel('Class Labels')
    plt.ylabel('Number of Points')
    plt.title('Class Distribution Across All .h5 Files')
    plt.xticks(classes)  # Ensure only class labels are shown on x-axis
    plt.show()

# Folder containing the .h5 files
folder_path = 'data/s3dis/buildings_h5_labels_fixed'

# Get class distribution
class_counts = get_class_distribution(folder_path)

# Plot class distribution
plot_class_distribution(class_counts)
