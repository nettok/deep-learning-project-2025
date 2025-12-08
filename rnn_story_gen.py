import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
from typing import List, Tuple, Dict
from collections import Counter
import matplotlib.pyplot as plt

def load_text_data(data_path: str) -> str:
    """
    Loads and concatenates text from all .txt files in the specified directory.

    Args:
        data_path (str): Path to the directory containing text files.

    Returns:
        str: The combined text content.
    """
    print(f"Loading text data from {data_path}...")
    text = ""
    try:
        files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
        if not files:
            raise FileNotFoundError(f"No .txt files found in {data_path}")
        
        for filename in sorted(files):
            file_path = os.path.join(data_path, filename)
            print(f"  - Reading {filename}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                text += f.read() + "\n"
        
        print(f"Total characters loaded: {len(text)}")
        return text
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        sys.exit(1)

def perform_eda(text: str):
    """
    Performs exploratory data analysis on the loaded text.
    """
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # 1. Basic Stats
    total_chars = len(text)
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    
    print(f"Total Characters: {total_chars}")
    print(f"Vocabulary Size:  {vocab_size}")
    print(f"Vocabulary:       {repr(''.join(vocab))}")
    
    # 2. Character Frequency
    print("-" * 40)
    print("Top 20 Most Common Characters:")
    counter = Counter(text)
    for char, count in counter.most_common(20):
        print(f"  {repr(char)}: {count} ({count/total_chars*100:.2f}%)")
        
    # 3. Head (First 500)
    print("-" * 40)
    print("First 500 Characters (Check for headers):")
    print("-" * 40)
    print(text[:500])
    print("-" * 40)

    # 4. Tail (Last 500)
    print("Last 500 Characters (Check for licenses/footers):")
    print("-" * 40)
    print(text[-500:])
    print("-" * 40)
    print("=" * 80)

def preprocess_text(text: str) -> Tuple[np.ndarray, List[str], Dict[str, int], np.ndarray, int]:
    """
    Vectorizes the text: creates vocabulary and mappings.

    Args:
        text (str): The input text.

    Returns:
        Tuple:
            - np.ndarray: Text converted to integer indices.
            - List[str]: The vocabulary (unique characters).
            - Dict[str, int]: Character to index mapping.
            - np.ndarray: Index to character mapping.
            - int: Vocabulary size.
    """
    print("Preprocessing text...")
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size} unique characters")

    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])
    
    return text_as_int, vocab, char2idx, idx2char, vocab_size

def create_dataset(text_as_int: np.ndarray, seq_length: int, batch_size: int, buffer_size: int) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset for training.

    Args:
        text_as_int (np.ndarray): The vectorized text.
        seq_length (int): Length of each sequence input.
        batch_size (int): Batch size.
        buffer_size (int): Buffer size for shuffling.

    Returns:
        tf.data.Dataset: The prepared dataset.
    """
    print(f"Creating dataset (seq_length={seq_length}, batch_size={batch_size})...")
    
    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    # Convert distinct characters to sequences of size seq_length+1
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)
    
    # Shuffle and batch
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def build_model(vocab_size: int, embedding_dim: int, rnn_units: int, batch_size: int, stateful: bool = False) -> keras.Model:
    """
    Builds the RNN model.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embedding layer.
        rnn_units (int): Number of RNN units.
        batch_size (int): Batch size.
        stateful (bool): Whether the GRU layer is stateful.

    Returns:
        keras.Model: The compiled model.
    """
    print(f"Building model (vocab_size={vocab_size}, rnn_units={rnn_units}, stateful={stateful})...")
    model = keras.Sequential([
        layers.Input(batch_shape=(batch_size, None)),
        layers.Embedding(vocab_size, embedding_dim),
        layers.GRU(rnn_units,
                   return_sequences=True,
                   stateful=stateful, 
                   recurrent_initializer='glorot_uniform'),
        layers.Dense(vocab_size)
    ])
    
    return model

def loss(labels, logits):
    """Computes the sparse categorical crossentropy loss."""
    return keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def train_model(model: keras.Model, dataset: tf.data.Dataset, epochs: int, checkpoint_dir: str, save_every_n_epochs: int = 3) -> keras.callbacks.History:
    """
    Trains the model.

    Args:
        model (keras.Model): The model to train.
        dataset (tf.data.Dataset): The training dataset.
        epochs (int): Number of epochs.
        checkpoint_dir (str): Directory to save checkpoints.
        save_every_n_epochs (int): Frequency of saving checkpoints (in epochs).

    Returns:
        keras.callbacks.History: Training history.
    """
    print(f"\nTraining RNN model (saving every {save_every_n_epochs} epochs)...")
    model.compile(optimizer='adam', loss=loss)

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Name of the checkpoint files template
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

    class PeriodCheckpoint(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % save_every_n_epochs == 0:
                # Format the filename with the current epoch (1-based)
                save_path = checkpoint_prefix.format(epoch=epoch+1)
                print(f"\nSaving checkpoint to {save_path}")
                self.model.save_weights(save_path)

    history = model.fit(dataset, epochs=epochs, callbacks=[PeriodCheckpoint()])
    return history

def generate_text(model: keras.Model, start_string: str, char2idx: Dict[str, int], idx2char: np.ndarray, num_generate: int = 1000, temperature: float = 1.0) -> str:
    """
    Generates text using the trained model.

    Args:
        model (keras.Model): The trained model.
        start_string (str): The seed text.
        char2idx (Dict[str, int]): Mapping from char to index.
        idx2char (np.ndarray): Mapping from index to char.
        num_generate (int): Number of characters to generate.
        temperature (float): Controls randomness. Low is predictable, high is surprising.

    Returns:
        str: The generated text.
    """
    print(f"\nGenerating {num_generate} characters with seed '{start_string}' (temp={temperature})...")

    # Vectorize start string
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    # Reset states for generation
    for layer in model.layers:
        if hasattr(layer, 'reset_states'):
            layer.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

def save_full_model(model: keras.Model, path: str):
    """Saves the entire model."""
    print(f"Saving model to {path}...")
    model.save(path)

def plot_training_history(history: keras.callbacks.History):
    """
    Plots the training loss.

    Args:
        history (keras.callbacks.History): The history object returned by model.fit().
    """
    print("\nPlotting training history...")
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    Main function to run the RNN story generator.

    Usage Modes:
    1. Train from scratch:
       python rnn_story_gen.py --author_dir data/rnn/oscar_wilde --epochs 50

    2. Resume training from checkpoint (e.g., epoch 5):
       python rnn_story_gen.py --author_dir data/rnn/oscar_wilde --load_weights_path rnn_checkpoints/ckpt_5.weights.h5 --epochs 45

    3. Generate text only (using saved full model):
       python rnn_story_gen.py --author_dir data/rnn/oscar_wilde --load_model_path final_model.keras --epochs 0

    Recommended Epochs:
    - fyodor_dostoyevsky: 15-30 epochs (larger dataset)
    - oscar_wilde: 40-60 epochs (smaller dataset)
    """
    parser = argparse.ArgumentParser(description="Train RNN for Story Generation")
    parser.add_argument('--author_dir', type=str, required=True, help="Directory containing author's text files (e.g., data/rnn/oscar_wilde)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--seq_length', type=int, default=100, help="Sequence length for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--embedding_dim', type=int, default=256, help="Embedding dimension")
    parser.add_argument('--rnn_units', type=int, default=1024, help="Number of RNN units")
    parser.add_argument('--save_model_path', type=str, help="Path to save the trained model (e.g., my_model.keras)")
    parser.add_argument('--seed_text', type=str, default="The", help="Seed text for generation")
    parser.add_argument('--generate_length', type=int, default=500, help="Length of text to generate after training")
    parser.add_argument('--temperature', type=float, default=1.0, help="Sampling temperature")
    parser.add_argument('--load_weights_path', type=str, help="Path to load pre-trained weights from (e.g., rnn_checkpoints/ckpt_3.weights.h5)")
    parser.add_argument('--load_model_path', type=str, help="Path to load a full pre-trained model from (e.g., my_model.keras)")
    parser.add_argument('--eda', action='store_true', help="Perform Exploratory Data Analysis (stats, head/tail check) and exit")

    args = parser.parse_args()

    # 1. Load Data
    text = load_text_data(args.author_dir)

    # 1.5. EDA (Optional)
    if args.eda:
        perform_eda(text)
        sys.exit(0)

    # 2. Preprocess
    text_as_int, vocab, char2idx, idx2char, vocab_size = preprocess_text(text)
    
    # 3. Create Dataset
    dataset = create_dataset(text_as_int, args.seq_length, args.batch_size, buffer_size=10000)

    # 4. Build or Load Model
    if args.load_model_path:
        print(f"Loading full model from {args.load_model_path}...")
        # We need to provide the custom loss function if it was saved with one, 
        # though usually sparse_categorical_crossentropy is standard.
        model = keras.models.load_model(args.load_model_path, custom_objects={'loss': loss})
    else:
        model = build_model(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            rnn_units=args.rnn_units,
            batch_size=args.batch_size
        )
        
        # Load weights if specified (only if we didn't load a full model)
        if args.load_weights_path:
            print(f"Loading weights from {args.load_weights_path}...")
            model.load_weights(args.load_weights_path)

    model.summary()

    # 5. Train (if epochs > 0)
    if args.epochs > 0:
        # Directory for checkpoints
        checkpoint_dir = './rnn_checkpoints'
        history = train_model(model, dataset, args.epochs, checkpoint_dir, save_every_n_epochs=3)
    else:
        print("Epochs set to 0. Skipping training.")

    # 6. Save Model (Optional)
    if args.save_model_path:
        # Note: Saving a model with custom input shapes (like the variable sequence length during generation)
        # can be tricky. We trained with fixed batch size.
        # For simple saving/loading to reuse weights, we can save weights or the whole model.
        save_full_model(model, args.save_model_path)

    # 7. Generate Text
    # To generate text, we need to rebuild the model with batch_size=1 to make it flexible for prediction
    # Or just use the trained model if we pass inputs correctly, but statefulness changes things.
    # The simplest way for this script's flow is to create a new model instance with batch_size=1
    # and load the weights we just trained.
    
    print("\nRebuilding model for generation (batch_size=1)...")
    model_gen = build_model(vocab_size, args.embedding_dim, args.rnn_units, batch_size=1, stateful=True)
    # Model is already built by Input layer
    model_gen.set_weights(model.get_weights())

    generated_story = generate_text(model_gen, args.seed_text, char2idx, idx2char, args.generate_length, args.temperature)
    
    print("-" * 80)
    print("GENERATED STORY:")
    print("-" * 80)
    print(generated_story)
    print("-" * 80)

if __name__ == "__main__":
    main()
