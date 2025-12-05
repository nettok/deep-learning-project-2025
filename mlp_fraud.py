import argparse
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from imblearn.combine import SMOTEENN
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple

# Suppress TensorFlow CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set random seed for reproducibility
SEED = 42
keras.utils.set_random_seed(SEED)

def load_creditcard_data(file_path: str) -> pd.DataFrame:
    """
    Loads the creditcard.csv dataset from the specified path.

    Args:
        file_path (str): The path to the creditcard.csv file.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    print(f"Attempting to load data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Successfully loaded data from {file_path}")
    return df

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Checks for missing values in the dataset.

    Args:
        df (pd.DataFrame): The dataset to check.

    Returns:
        pd.Series: A series containing the count of missing values for each column.
    """
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values found:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found in the dataset.")
    return missing_values

def plot_class_distribution(df: pd.DataFrame, target_col: str = 'Class', show_plot: bool = True) -> None:
    """
    Plots the distribution of the target class.

    Args:
        df (pd.DataFrame): The dataset.
        target_col (str): The name of the target column.
        show_plot (bool): Whether to display the plot.
    """
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in dataset.")

    counts = df[target_col].value_counts()
    print(f"Class counts:\n{counts}")
    print(f"Fraud percentage: {counts[1] / len(df) * 100:.4f}%")

    if show_plot:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=target_col, data=df)
        plt.title('Class Distribution')
        plt.xlabel('Class (0: Legitimate, 1: Fraudulent)')
        plt.ylabel('Count')
        plt.show()

def plot_correlation_matrix(df: pd.DataFrame, show_plot: bool = True) -> None:
    """
    Plots the correlation matrix of the dataset.

    Args:
        df (pd.DataFrame): The dataset.
        show_plot (bool): Whether to display the plot.
    """
    if show_plot:
        plt.figure(figsize=(12, 10))
        corr = df.corr()
        sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size': 20})
        plt.title('Correlation Matrix', fontsize=14)
        plt.show()

def drop_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Drops the 'Time' column from the dataset."""
    if 'Time' in df.columns:
        print("Dropping 'Time' column...")
        return df.drop(columns=['Time'])
    return df

def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scales the 'Amount' column using StandardScaler fitted on training data to prevent leakage.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        X_test (pd.DataFrame): Test features.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The scaled DataFrames.
    """
    if 'Amount' in X_train.columns:
        print("Scaling 'Amount' column...")
        scaler = StandardScaler()
        
        # Use .copy() to avoid SettingWithCopyWarning
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()
        
        # Fit on training data ONLY
        X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
        
        # Transform validation and test data
        X_val['Amount'] = scaler.transform(X_val[['Amount']])
        X_test['Amount'] = scaler.transform(X_test[['Amount']])
        
    return X_train, X_val, X_test

def split_data(df: pd.DataFrame, target_col: str = 'Class', test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and test sets using stratification.
    """
    print(f"Splitting data (test_size={test_size})...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)

def apply_resampling(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies resampling (SMOTE+ENN) to the training data.
    """
    print("Applying SMOTE+ENN (Hybrid Resampling)...")
    smote_enn = SMOTEENN(random_state=SEED)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    
    print(f"New training counts: {y_resampled.value_counts()}")
    
    return X_resampled, y_resampled

def build_mlp_model(input_dim: int) -> keras.Model:
    """
    Builds and compiles a Multi-Layer Perceptron (MLP) model for binary classification.

    Args:
        input_dim (int): The number of input features.

    Returns:
        keras.Model: The compiled MLP model.
    """
    model = keras.Sequential([
        layers.InputLayer(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[
                      'accuracy',
                      keras.metrics.Precision(name='precision'),
                      keras.metrics.Recall(name='recall'),
                  ])
    return model

def train_mlp_model(model: keras.Model, X_train, y_train, X_val, y_val, epochs: int = 50, batch_size: int = 256) -> keras.callbacks.History:
    """
    Trains the MLP model with early stopping and learning rate reduction.
    """
    print("\nTraining MLP model...")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    ]

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=1)
    return history

def plot_training_history(history: keras.callbacks.History, show_plot: bool = True) -> None:
    """
    Plots the training history (accuracy, loss).

    Args:
        history: The history object returned by model.fit().
        show_plot (bool): Whether to display the plot.
    """
    if show_plot:
        metrics = ['accuracy', 'loss']
        plt.figure(figsize=(12, 5))

        for i, metric in enumerate(metrics):
            plt.subplot(1, len(metrics), i + 1)
            plt.plot(history.history[metric], label=f'Train {metric.capitalize()}')
            plt.plot(history.history[f'val_{metric}'], label=f'Val {metric.capitalize()}')
            plt.title(f'Model {metric.capitalize()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()

        plt.tight_layout()
        plt.show()

def find_optimal_threshold(model: keras.Model, X_test, y_test, beta: float = 1.0, show_plot: bool = True) -> float:
    """
    Finds the optimal threshold that maximizes the F-beta score.

    Args:
        model (keras.Model): The trained model.
        X_test: Test features.
        y_test: Test labels.
        beta (float): The beta value for the F-beta score.
            beta=1.0 favors precision and recall equally (F1 score).
            beta=2.0 favors recall more than precision (F2 score).
            beta=0.5 favors precision more than recall.
        show_plot (bool): Whether to display the plot.

    Returns:
        float: The optimal threshold.
    """
    print(f"\nFinding optimal threshold (beta={beta})...")
    y_pred_prob = model.predict(X_test, verbose=0)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
    
    # Calculate F-beta score for each threshold
    f_scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls)
    
    # Handle potential division by zero (resulting in NaNs)
    f_scores = np.nan_to_num(f_scores)
    
    # Find the index of the maximum F-beta score
    best_idx = np.argmax(f_scores)
    best_threshold = thresholds[best_idx]
    best_score = f_scores[best_idx]
    
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"Best F{beta} Score: {best_score:.4f}")
    
    if show_plot:
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker='.', label='MLP')
        plt.scatter(recalls[best_idx], precisions[best_idx], marker='o', color='red', label=f'Best Threshold (beta={beta})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (Optimizing F{beta})')
        plt.legend()
        plt.show()
    
    return best_threshold

def evaluate_model(model: keras.Model, X_test, y_test, threshold: float = 0.5, show_plot: bool = True) -> None:
    """
    Evaluates the model on the test set and plots the confusion matrix.
    """
    print(f"\nEvaluating model on test set (Threshold: {threshold:.4f})...")
    results = model.evaluate(X_test, y_test, verbose=0)
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > threshold).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if show_plot:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix (Threshold={threshold:.4f})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train MLP for Credit Card Fraud Detection")
    parser.add_argument('--data_path', type=str, default="data/mlp/raw/creditcard.csv", help="Path to creditcard.csv")
    parser.add_argument('--no_plots', action='store_true', help="Disable plotting")
    args = parser.parse_args()

    show_plots = not args.no_plots
    
    try:
        credit_card_df = load_creditcard_data(args.data_path)
        print(credit_card_df.head())
        print(f"Dataset shape: {credit_card_df.shape}")
        
        check_missing_values(credit_card_df)
        plot_class_distribution(credit_card_df, show_plot=show_plots)
        plot_correlation_matrix(credit_card_df, show_plot=show_plots)
        
        # Preprocessing
        df_clean = drop_time_column(credit_card_df)
        
        # Splitting (Train/Test)
        X_train_full, X_test, y_train_full, y_test = split_data(df_clean)

        # Further split training data into training and validation sets (Validation remains pure/imbalanced)
        print("Splitting training data into partial train and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=SEED)
        
        # Scaling (Fit on Train, Transform Val and Test)
        X_train, X_val, X_test = scale_features(X_train, X_val, X_test)
        
        # Resampling (Only on the partial training set)
        X_train_res, y_train_res = apply_resampling(X_train, y_train)
        
        # Build and compile the MLP model
        input_dim = X_train_res.shape[1]
        mlp_model = build_mlp_model(input_dim)
        mlp_model.summary()
        
        # Train the model
        history = train_mlp_model(mlp_model, X_train_res, y_train_res, X_val, y_val)
        
        # Plot training history
        plot_training_history(history, show_plot=show_plots)
        
        # Evaluate on test set
        optimal_threshold = find_optimal_threshold(mlp_model, X_test, y_test, beta=2, show_plot=show_plots)
        evaluate_model(mlp_model, X_test, y_test, threshold=optimal_threshold, show_plot=show_plots)
        
    except FileNotFoundError:
        print(f"Error: The file {args.data_path} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

if __name__ == "__main__":
    main()
