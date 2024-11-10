# scripts/convert_dota_annotations.py

import os
import json
import torch
from ops import polygon_to_obb
from tqdm import tqdm

def convert_pxxx_to_obb(pxxx_path, obb_path):
    """
    Convert DOTA Pxxx.txt annotation file to OBB format JSON.

    Args:
        pxxx_path (str): Path to the Pxxx.txt file.
        obb_path (str): Path to save the converted OBB JSON.
    """
    obb_annotations = []
    with open(pxxx_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue  # Invalid annotation line
            class_name = parts[0]
            coords = list(map(float, parts[1:9]))  # x1 y1 x2 y2 x3 y3 x4 y4
            obb = polygon_to_obb(coords)  # Returns [x_center, y_center, w, h, qx, qy, qz, qw]
            obb_annotations.append({
                'category': class_name,
                'obb': obb
            })
    
    with open(obb_path, 'w') as f:
        json.dump({'annotations': obb_annotations}, f)

def main():
    train_pxxx_dir = 'data/DOTA/train/annotations/'
    val_pxxx_dir = 'data/DOTA/val/annotations/'
    
    train_output_dir = 'data/DOTA/train/annotations_converted/'
    val_output_dir = 'data/DOTA/val/annotations_converted/'
    
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)
    
    # Convert training annotations
    train_pxxx_files = [f for f in os.listdir(train_pxxx_dir) if f.endswith('.txt')]
    print("Converting DOTA training annotations...")
    for pxxx_file in tqdm(train_pxxx_files):
        pxxx_path = os.path.join(train_pxxx_dir, pxxx_file)
        obb_filename = pxxx_file.replace('.txt', '.json')
        obb_path = os.path.join(train_output_dir, obb_filename)
        convert_pxxx_to_obb(pxxx_path, obb_path)
    
    # Convert validation annotations
    val_pxxx_files = [f for f in os.listdir(val_pxxx_dir) if f.endswith('.txt')]
    print("Converting DOTA validation annotations...")
    for pxxx_file in tqdm(val_pxxx_files):
        pxxx_path = os.path.join(val_pxxx_dir, pxxx_file)
        obb_filename = pxxx_file.replace('.txt', '.json')
        obb_path = os.path.join(val_output_dir, obb_filename)
        convert_pxxx_to_obb(pxxx_path, obb_path)
    
    print("DOTA annotations conversion completed.")

if __name__ == "__main__":
    main()
