import matplotlib.pyplot as plt
import numpy as np
from typing import Any

def visualize_loss(history, bin):
    # Visualize the training and validation results
    accuracy = [res['acc'] for res in history]
    losses = [res['loss'] for res in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
    ax1.plot(losses, '-o', label = 'Loss')
    ax1.set_xlabel("Number of Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(100 * np.array(accuracy), '-o', label = 'Accuracy')
    ax2.set_xlabel("Number of Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend();

    fig.show()
    plt.savefig(f'./data/plot_images/train_loss_curves/{bin}.jpg', bbox_inches='tight')

