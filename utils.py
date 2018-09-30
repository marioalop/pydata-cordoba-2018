# Utility functions
import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_loss_and_accuracy(model_history, acc_metric):
    plt.figure(figsize=(18,6))

    # Loss Curves
    plt.subplot(1,2,1)
    plt.plot(model_history.history['loss'], color='#3c8fb9', linewidth=2, linestyle='dotted')
    plt.plot(model_history.history['val_loss'], color='#f47a50', linewidth=3)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # Accuracy Curves
    plt.subplot(1,2,2)
    plt.plot(model_history.history[acc_metric], color='#3c8fb9', linewidth=2, linestyle='dotted')
    plt.plot(model_history.history['val_' + acc_metric], color='#f47a50', linewidth=3)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy (' + acc_metric + ') Curves', fontsize=16)

    plt.show()