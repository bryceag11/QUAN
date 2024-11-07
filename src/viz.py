# viz.py

import matplotlib.pyplot as plt

def plot_losses(epochs, total_loss, bbox_loss, class_loss, quat_loss, save_path='loss_plot.png'):
    """
    Plots the Total Loss, Bounding Box Loss, Classification Loss, and Quaternion Loss against epochs.
    
    Args:
        epochs (list or array-like): List of epoch numbers.
        total_loss (list or array-like): List of total loss values per epoch.
        bbox_loss (list or array-like): List of bounding box loss values per epoch.
        class_loss (list or array-like): List of classification loss values per epoch.
        quat_loss (list or array-like): List of quaternion loss values per epoch.
        save_path (str): Path to save the generated plot image.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, total_loss, label='Total Loss', marker='o')
    plt.plot(epochs, bbox_loss, label='Bounding Box Loss', marker='s')
    plt.plot(epochs, class_loss, label='Classification Loss', marker='^')
    plt.plot(epochs, quat_loss, label='Quaternion Loss', marker='d')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()
    
    print(f"Loss plot saved to {save_path}")
