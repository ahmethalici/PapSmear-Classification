# src/visualization/visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
import matplotlib as mpl


def plot_confusion_matrix(y_true, y_pred, class_names, title):
    """
    Generates and displays a confusion matrix plot.
    """

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

def plot_training_history(history, title):
    """
    Plots training and validation accuracy and loss.
    """
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def get_grad_model(model, backbone_name, last_conv_layer_name):
    """
    Creates a new Keras Model that outputs the last convolutional layer's
    activations and the final prediction.
    """
    try:
        backbone = model.get_layer(backbone_name)
        last_conv_output = backbone.get_layer(last_conv_layer_name).output
        backbone_output = backbone.output

        head_layers = []
        found_backbone = False
        for layer in model.layers:
            if layer.name == backbone_name:
                found_backbone = True
                continue
            if found_backbone:
                head_layers.append(layer)

        x = backbone_output
        for layer in head_layers:
            x = layer(x)
        final_prediction = x

        grad_model = tf.keras.Model(
            inputs=backbone.inputs,
            outputs=[last_conv_output, final_prediction]
        )
        grad_model.trainable = False
        return grad_model
    except Exception as e:
        print(f"Error creating Grad-CAM model: {e}")
        print("Please ensure `backbone_name` and `last_conv_layer_name` in config.yml are correct for the selected model.")
        return None

def generate_gradcam_heatmap(img_batch, grad_model):
    """Generates a Grad-CAM heatmap for a given image batch and model."""
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch, training=False)
        # Use the prediction for the positive class
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    
    # Correct einsum for batch size of 1
    heatmap = tf.einsum("hwc,c->hw", conv_outputs[0], pooled_grads[0])
    
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(img_array, heatmap, alpha=0.6):
    """Overlays a heatmap onto an image."""
    heatmap_resized = tf.image.resize(np.expand_dims(heatmap, -1), img_array.shape[:2]).numpy()
    heatmap_jet = np.uint8(255 * heatmap_resized.squeeze())
    jet_colors = mpl.colormaps["jet"](np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_jet]
    
    jet_heatmap_img = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap_array = tf.keras.utils.img_to_array(jet_heatmap_img)
    
    overlaid_img = jet_heatmap_array * alpha + img_array
    overlaid_img = tf.keras.utils.array_to_img(overlaid_img)
    return overlaid_img
