import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


def load_raw_data(split_ratio: float = 0.8, data_dir: str = 'data/cnn/') -> Tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    """
    Loads the raw Plant Village dataset from TensorFlow Datasets.
    """
    print(f"Loading Plant Village dataset into {data_dir}...")
    
    train_split = f'train[:{int(split_ratio*100)}%]'
    val_split = f'train[{int(split_ratio*100)}%:]'

    (train_ds, val_ds), info = tfds.load(
        'plant_village',
        split=[train_split, val_split],
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
        data_dir=data_dir
    )

    print(f"Total classes: {info.features['label'].num_classes}")
    print(f"Class names: {info.features['label'].names}")
    
    return train_ds, val_ds, info

def preprocess_dataset(ds: tf.data.Dataset, is_training: bool, batch_size: int, img_size: int) -> tf.data.Dataset:
    """
    Applies preprocessing (resizing, rescaling, augmentation) and batching to the dataset.
    """
    # Preprocessing layers
    resize_and_rescale = keras.Sequential([
        layers.Resizing(img_size, img_size),
        layers.Rescaling(1./255)
    ])

    # Resize and Rescale
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.shuffle(1000)

    # Batch the dataset
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set
    if is_training:
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
        ])
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Use buffered prefetching
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

def visualize_data(dataset: tf.data.Dataset, class_names: list, show_plot: bool = True):
    """
    Visualizes a few samples from the dataset.
    """
    if not show_plot:
        return

    print("Visualizing data samples...")
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy()) # Images are already rescaled 0-1
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

def perform_eda(dataset: tf.data.Dataset, class_names: list, show_plot: bool = True):
    """
    Performs Exploratory Data Analysis (EDA) on the dataset.
    Specifically checks for class imbalance.
    """
    if not show_plot:
        return

    print("Performing EDA (calculating class distribution)...")
    
    # Initialize counts
    counts = np.zeros(len(class_names), dtype=int)
    
    # Iterate over the dataset to count labels
    for _, labels in dataset:
        unique, counts_batch = np.unique(labels.numpy(), return_counts=True)
        counts[unique] += counts_batch
            
    # Sort by count
    indices = np.argsort(counts)[::-1]
    sorted_names = [class_names[i] for i in indices]
    sorted_counts = counts[indices]

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_names, sorted_counts)
    plt.xlabel('Class Names')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Training Set')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    print("Top 5 classes by count:")
    for i in range(min(5, len(sorted_names))):
        print(f"  {sorted_names[i]}: {sorted_counts[i]}")

def build_cnn_model(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    """
    Builds a Convolutional Neural Network (CNN) model.

    Args:
        input_shape (Tuple[int, int, int]): Shape of input images.
        num_classes (int): Number of target classes.

    Returns:
        keras.Model: The compiled CNN model.
    """
    print(f"Building CNN model (input_shape={input_shape}, num_classes={num_classes})...")
    
    model = keras.Sequential([
        layers.InputLayer(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return model

def train_model(model: keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, epochs: int, checkpoint_path: str = None) -> keras.callbacks.History:
    """
    Trains the CNN model.
    """
    print("\nTraining CNN model...")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]

    if checkpoint_path:
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
        callbacks.append(checkpoint_cb)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history

def plot_history(history: keras.callbacks.History, show_plot: bool = True):
    """
    Plots training and validation accuracy and loss.
    """
    if not show_plot:
        return

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.show()

def evaluate_model_performance(model: keras.Model, dataset: tf.data.Dataset, class_names: list, show_plot: bool = True):
    """
    Evaluates the model using classification report and confusion matrix.
    """
    print("\nCalculating detailed evaluation metrics...")
    
    y_true = []
    y_pred = []

    # Iterate once to ensure predictions and labels are aligned
    for images, labels in dataset:
        y_true.extend(labels.numpy())
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    if show_plot:
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

def evaluate_predictions(model: keras.Model, val_ds: tf.data.Dataset, class_names: list, show_plot: bool = True):
    """
    Visualizes some predictions.
    """
    if not show_plot:
        return

    print("\nVisualizing predictions...")
    plt.figure(figsize=(10, 10))
    for images, labels in val_ds.take(1):
        predictions = model.predict(images)
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy()) # Images are already rescaled 0-1
            
            predicted_label = class_names[np.argmax(predictions[i])]
            actual_label = class_names[labels[i]]
            
            confidence = 100 * np.max(predictions[i])
            
            plt.title(f"Act: {actual_label}\nPred: {predicted_label} ({confidence:.2f}%)", fontsize=10)
            plt.axis("off")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train CNN for Plant Disease Classification")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--img_size', type=int, default=256, help="Image size (height/width)")
    parser.add_argument('--data_dir', type=str, default="data/cnn/", help="Directory to download/load dataset")
    parser.add_argument('--checkpoint_dir', type=str, default="cnn_checkpoints", help="Directory to save checkpoints")
    parser.add_argument('--model_save_path', type=str, default="cnn_plant_health_model.keras", help="Path to save the final model")
    parser.add_argument('--load_model', type=str, help="Path to load a full model (.keras) instead of training")
    parser.add_argument('--load_weights', type=str, help="Path to load weights (.h5) into the model instead of training")
    parser.add_argument('--no_plots', action='store_true', help="Disable plotting")
    args = parser.parse_args()

    show_plots = not args.no_plots

    try:
        # Load Data
        train_ds, val_ds, info = load_raw_data(split_ratio=0.8, data_dir=args.data_dir)
        num_classes = info.features['label'].num_classes
        class_names = info.features['label'].names
        input_shape = (args.img_size, args.img_size, 3)

        # Perform EDA on raw data (Training Mode only)
        if not args.load_model and not args.load_weights:
            perform_eda(train_ds, class_names, show_plot=show_plots)
        
        # Preprocess Data
        train_ds = preprocess_dataset(train_ds, is_training=True, batch_size=args.batch_size, img_size=args.img_size)
        val_ds = preprocess_dataset(val_ds, is_training=False, batch_size=args.batch_size, img_size=args.img_size)
        
        model = None

        if args.load_model:
            print(f"Loading model from {args.load_model}...")
            model = keras.models.load_model(args.load_model)
        elif args.load_weights:
            print(f"Building model and loading weights from {args.load_weights}...")
            model = build_cnn_model(input_shape, num_classes)
            model.load_weights(args.load_weights)
        else:
            # Training Mode
            visualize_data(train_ds, class_names, show_plot=show_plots)
            
            model = build_cnn_model(input_shape, num_classes)
            model.summary()

            # Checkpoint setup
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            checkpoint_path = os.path.join(args.checkpoint_dir, "best.weights.h5")

            # Train
            history = train_model(model, train_ds, val_ds, epochs=args.epochs, checkpoint_path=checkpoint_path)

            # Save Final Model
            print(f"Saving final model to {args.model_save_path}...")
            model.save(args.model_save_path)

            # Plot History
            plot_history(history, show_plot=show_plots)
        
        # Evaluate/Visualize (runs for both training and loading modes)
        if model:
            # Compile isn't strictly necessary for predict() but good for evaluate() if we added that.
            # If loaded from weights, it might not be compiled.
            if args.load_weights:
                 model.compile(
                    optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy']
                )

            evaluate_model_performance(model, val_ds, class_names, show_plot=show_plots)
            evaluate_predictions(model, val_ds, class_names, show_plot=show_plots)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
