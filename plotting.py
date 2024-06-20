import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(train_losses, valid_losses, test_losses, valid_accuracies, test_accuracies):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot Losses
    for losses, label, color in zip([train_losses, valid_losses, test_losses],
                                    ['Training Loss', 'Validation Loss', 'Testing Loss'],
                                    ['blue', 'green', 'red']):
        axes[0].plot(np.mean(losses, axis=1), label=label, color=color)

    # Plot Accuracies
    axes[1].plot(valid_accuracies, label='Validation Accuracy', color='green')
    axes[1].plot(test_accuracies, label='Testing Accuracy', color='red')

    for ax in axes:
        ax.set_xlabel('Epochs')
        ax.set_title('Metrics')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
