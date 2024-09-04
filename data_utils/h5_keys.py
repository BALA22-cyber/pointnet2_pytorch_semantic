import h5py
import numpy as np

with h5py.File('/mnt/e/pointnet2_pytorch_semantic/data/s3dis/buildings_h5_labels/B1505_NE_2all_processed_data.h5', 'r') as f:
    processed_data = f['processed_data']

    # Get the raw bytes of the block0_values dataset
    block0_values_raw = processed_data['block0_values'][:]

    # Determine the data type based on the information from h5dump
    # For example, if the data type is a custom 64-bit float
    block0_values = np.frombuffer(block0_values_raw, dtype=np.float64).reshape((2697361, 8))

    print(f"block0_values shape: {block0_values.shape}")
    print(f"block0_values dtype: {block0_values.dtype}")