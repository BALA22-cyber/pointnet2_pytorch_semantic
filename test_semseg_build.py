import argparse
import os
import torch
import sys
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
import importlib
from torch.utils.data import DataLoader
from data_utils.BuildingDataLoader import BuildingDatasetWholeScene  # Import your dataset class

# Define color mapping for visualization (window, wall, door, others)
colors = {
    0: (255, 0, 0),    # window: red
    1: (0, 255, 0),    # wall: green
    2: (0, 0, 255),    # door: blue
    3: (255, 255, 0)   # others: yellow
}

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='Specify GPU device')
    parser.add_argument('--num_point', type=int, default=4096, help='Number of points per block [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment log directory')
    parser.add_argument('--num_votes', type=int, default=1, help='Number of votes to aggregate predictions [default: 1]')
    parser.add_argument('--visual', action='store_true', default=False, help='Save .obj files for visualization [default: False]')
    return parser.parse_args()

def setup_logger(experiment_dir):
    """ Setup logger to log both to a file and console output """
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    
    # Log to a file
    file_handler = logging.FileHandler(os.path.join(experiment_dir, 'eval.txt'))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Log to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_predictions_to_obj(points, pred_labels, output_filename, colors=None):
    """
    Save the predicted points and labels to an .obj file for visualization.
    Args:
        points (np.array): Array of points (Nx6 for XYZRGB).
        pred_labels (np.array): Predicted labels for each point (Nx1).
        output_filename (str): Output path for the .obj file.
        colors (dict): Optional dictionary of label to RGB color mapping.
    """
    with open(output_filename, 'w') as fout:
        for i in range(points.shape[0]):
            x, y, z = points[i, 0], points[i, 1], points[i, 2]
            if colors is not None and pred_labels[i] in colors:
                r, g, b = colors[pred_labels[i]]
            else:
                r, g, b = 255, 255, 255  # Default color is white
            fout.write(f"v {x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

def save_groundtruth_to_obj(points, gt_labels, output_filename, colors=None):
    """
    Save the ground truth points and labels to an .obj file for visualization.
    Args:
        points (np.array): Array of points (Nx6 for XYZRGB).
        gt_labels (np.array): Ground truth labels for each point (Nx1).
        output_filename (str): Output path for the .obj file.
        colors (dict): Optional dictionary of label to RGB color mapping.
    """
    with open(output_filename, 'w') as fout:
        for i in range(points.shape[0]):
            x, y, z = points[i, 0], points[i, 1], points[i, 2]
            if colors is not None and gt_labels[i] in colors:
                r, g, b = colors[gt_labels[i]]
            else:
                r, g, b = 255, 255, 255  # Default color is white
            fout.write(f"v {x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    """
    Aggregate votes from multiple predictions.
    Args:
        vote_label_pool (np.array): Voting array to accumulate predictions.
        point_idx (np.array): Indices of the points in the current batch.
        pred_label (np.array): Predicted labels for the current batch.
        weight (np.array): Weights for each prediction.
    """
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0:
                vote_label_pool[b, n, int(pred_label[b, n])] += 1  # Correcting the indexing here
    return vote_label_pool

def load_model_from_file(model_filepath):
    """ Dynamically load the model from a Python file """
    spec = importlib.util.spec_from_file_location("pointnet2_sem_seg", model_filepath)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return model_module


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = Path(experiment_dir) / 'visual'
    visual_dir.mkdir(exist_ok=True)
    
    # Setup the logger
    logger = setup_logger(experiment_dir)
    
    def log_string(msg):
        logger.info(msg)
        print(msg)
    
    log_string('---- EVALUATION START ----')

    # # Load the model dynamically based on the logs
    # # model_name = os.listdir(os.path.join(experiment_dir, 'logs'))[0].split('.')[0]
    # model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    # MODEL = importlib.import_module(model_name)

    # classifier = MODEL.get_model(4)  # 4 classes (window, wall, door, others)
    # classifier.cuda()


    # Add the models directory to the Python path
    model_dir = '/mnt/e/pointnet2_pytorch_semantic/models'  # Path to the models folder
    sys.path.append(model_dir)  # Add this directory to the Python path
    logger.info(f"Model directory added to path: {model_dir}")

    # Hardcode the model name (since we know it's pointnet2_sem_seg.py)
    model_name = 'pointnet2_sem_seg'

    # Try to import the model
    try:
        MODEL = importlib.import_module(model_name)
        logger.info(f"Successfully imported module: {model_name}")
    except ModuleNotFoundError as e:
        logger.error(f"Module {model_name} not found in {model_dir}")
        raise e

    classifier = MODEL.get_model(4)  # 4 classes (window, wall, door, others)
    classifier.cuda()

    # Load the model weights
    # checkpoint = torch.load(os.path.join(experiment_dir, 'checkpoints', 'best_model.pth'))
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    classifier.eval()

    # Load the test dataset
    dataset = BuildingDatasetWholeScene(root='/mnt/e/pointnet2_pytorch_semantic/data/s3dis/buildings_h5_4_labels_z_rotated_y_mod_downsamp_0.2', split='test', block_points=args.num_point)
    log_string(f"Number of test data points: {len(dataset)}")
    # DataLoader for testing
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    total_seen_class = [0 for _ in range(4)]  # 4 classes
    total_correct_class = [0 for _ in range(4)]
    total_iou_deno_class = [0 for _ in range(4)]

    # Directory to save .obj files
    pred_save_dir = os.path.join(experiment_dir, 'predictions')
    gt_save_dir = os.path.join(experiment_dir, 'groundtruth')
    if args.visual:
        os.makedirs(pred_save_dir, exist_ok=True)
        os.makedirs(gt_save_dir, exist_ok=True)

    # Start testing
    for batch_idx, (points, gt_labels) in enumerate(tqdm(test_loader, desc="Testing", unit="batch")):
        points = points.cuda().transpose(2, 1)  # [B, N, 6] to [B, 6, N]
        gt_labels = gt_labels.long().cuda()

        # Voting mechanism for aggregation
        vote_label_pool = np.zeros((points.shape[0], args.num_point, 4))

        for vote_idx in range(args.num_votes):
            with torch.no_grad():
                seg_pred, _ = classifier(points)
                pred_labels = seg_pred.argmax(dim=2).cpu().numpy()  # Predicted labels

            vote_label_pool = add_vote(vote_label_pool, np.arange(pred_labels.shape[0]), pred_labels, np.ones_like(pred_labels))

        # Get final predictions by majority voting
        final_pred_labels = np.argmax(vote_label_pool, axis=2)

        points = points.cpu().numpy().transpose(0, 2, 1)  # Back to [B, N, 6]
        gt_labels = gt_labels.cpu().numpy()

        # Save predictions and ground truth as .obj files for visualization (if visual flag is set)
        if args.visual:
            for b in range(points.shape[0]):
                # Save predictions
                output_filename_pred = os.path.join(pred_save_dir, f"batch_{batch_idx}_sample_{b}_pred.obj")
                save_predictions_to_obj(points[b], final_pred_labels[b], output_filename_pred, colors)

                # Save ground truth
                output_filename_gt = os.path.join(gt_save_dir, f"batch_{batch_idx}_sample_{b}_gt.obj")
                save_groundtruth_to_obj(points[b], gt_labels[b], output_filename_gt, colors)

        # Compute accuracy and IoU metrics
        for l in range(4):
            total_seen_class[l] += np.sum((gt_labels == l))
            total_correct_class[l] += np.sum((final_pred_labels == l) & (gt_labels == l))
            total_iou_deno_class[l] += np.sum(((final_pred_labels == l) | (gt_labels == l)))

    # Calculate IoU and accuracy
    IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
    for l in range(4):
        logger.info(f'Class {l} IoU: {IoU[l]:.4f}')
    logger.info(f'Overall IoU: {np.mean(IoU):.4f}')
    logger.info(f'Overall accuracy: {np.sum(total_correct_class) / np.sum(total_seen_class):.4f}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
