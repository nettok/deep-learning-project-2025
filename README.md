# Deep Learning Project 2025

**Author:** Ernesto Menéndez

This project implements three different deep learning models to solve distinct tasks: **Binary Classification (Fraud Detection)**, **Image Classification (Plant Disease)**, and **Text Generation (Story Generation)**.

## 1. MLP - Credit Card Fraud Detection

[Link to Notebook](01_MLP.ipynb)

### Dataset
*   **Source:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle).
*   **Description:** Contains ~285,000 credit card transactions from September 2013.
*   **Challenge:** Highly imbalanced dataset with only 0.172% fraudulent transactions.

### Model
*   **Architecture:** Multi-Layer Perceptron (MLP) with 2 hidden layers (128 and 64 units).
*   **Key Techniques:**
    *   **Batch Normalization & Dropout:** For regularization and faster convergence.
    *   **SMOTE+ENN:** Hybrid resampling technique applied to training data to handle class imbalance.
    *   **Threshold Tuning:** Optimized decision threshold based on F2-Score to prioritize Recall.

### Main Results
*   **Accuracy:** ~99.9%
*   **Best F2-Score:** ~0.83 (at optimized threshold)
*   **Outcome:** High recall for fraud cases, effectively identifying fraudulent transactions while maintaining a reasonable precision.

### How to Run
```bash
python mlp_fraud.py --data_path data/mlp/raw/creditcard.csv
```

---

## 2. CNN - Plant Disease Classification

[Link to Notebook](02_CNN.ipynb)

### Dataset
*   **Source:** [Plant Village](https://www.tensorflow.org/datasets/catalog/plant_village) (TensorFlow Datasets).
*   **Description:** ~54,000 images of plant leaves categorized into 38 classes (species + disease/healthy status).

### Model
*   **Architecture:** Convolutional Neural Network (CNN) with 5 Convolutional blocks.
*   **Key Techniques:**
    *   **Layers:** Conv2D (32/64 filters) + MaxPooling2D + ReLU activation.
    *   **Data Augmentation:** Random flips and rotations to prevent overfitting.
    *   **Callbacks:** EarlyStopping and ReduceLROnPlateau.

### Main Results
*   **Validation Accuracy:** ~93-94%
*   **Outcome:** The model successfully learns to distinguish between 38 different plant disease categories with high confidence.

### How to Run
```bash
python cnn_plant_health.py --epochs 10
```

---

## 3. RNN - Story Generation

[Link to Notebook](03_RNN.ipynb)

### Dataset
*   **Source:** [Project Gutenberg](https://www.gutenberg.org/).
*   **Description:** Plaintext books from **Fyodor Dostoyevsky** (~3MB) and Oscar Wilde. Cleaned to remove headers/footers.

### Model
*   **Architecture:** Recurrent Neural Network (RNN).
*   **Key Techniques:**
    *   **Embedding Layer:** Maps characters to a dense vector space.
    *   **GRU (Gated Recurrent Unit):** 1024 units to capture temporal dependencies (chosen over LSTM for efficiency).
    *   **Character-level Generation:** Predicts the next character in a sequence.

### Main Results
*   **Final Loss:** ~0.87
*   **Outcome:** Successfully mimics the author's writing style, vocabulary, and grammar. However, generated text lacks long-term semantic coherence (hallucinations).

### How to Run
```bash
# Train
python rnn_story_gen.py --author_dir data/rnn/fyodor_dostoyevsky --epochs 20

# Generate (using pre-trained weights from epoch 18)
python rnn_story_gen.py --author_dir data/rnn/fyodor_dostoyevsky --load_weights_path rnn_checkpoints/ckpt_18.weights.h5 --epochs 0
```

---

## Environment Setup

1.  **Create Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Datasets:**
    *   **MLP:** Download `creditcard.csv` from Kaggle and place it in `data/mlp/raw/`. Source: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
    *   **RNN:** Download plain text books (e.g., Dostoyevsky) from Project Gutenberg and place them in `data/rnn/fyodor_dostoyevsky/` (or your chosen author folder).
    *   **CNN:** No action needed. The dataset is automatically downloaded via TensorFlow Datasets.

## Conclusions
*   **Imbalance Handling:** Hybrid resampling (SMOTE+ENN) and threshold moving are critical for fraud detection tasks.
*   **CNN Effectiveness:** Even simple custom CNN architectures can achieve high accuracy on complex image classification tasks given sufficient data and augmentation.
*   **RNN Limitations:** While character-level GRUs capture style effectively, they struggle with long-term narrative consistency, suggesting the need for Transformer-based architectures for better coherence.
