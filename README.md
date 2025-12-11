# Sentiment Analysis with LSTM

A PyTorch-based sentiment analysis project using Long Short-Term Memory (LSTM) networks for text classification. The model classifies text into three sentiment categories: Negative, Neutral, and Positive.

## Project Overview

This project implements a complete end-to-end sentiment analysis pipeline, from data preprocessing to model training and inference. The model uses bidirectional LSTM layers for capturing context in both directions, achieving robust sentiment classification performance.

## Project Structure

```
Sentiment_LSTM/
├── PreProcessing.ipynb              # Data cleaning and preprocessing notebook
├── SentimentAnaylsis_LSTM.ipynb     # Main training notebook with model architecture
├── Sentiment_LSTM.py                # Standalone inference script
├── requirements.txt                 # Project dependencies
├── vocab.pkl                        # Saved vocabulary dictionary
├── sentiment_model.pkl              # Trained model weights
└── README.md                        # This file
```

## Features

- **Multiple RNN Architectures**: Supports RNN, LSTM, and GRU models
- **Bidirectional Processing**: Captures context from both directions for better understanding
- **Text Preprocessing**: Comprehensive cleaning including URL removal, mention filtering, and tokenization
- **Vocabulary Management**: Custom vocabulary with special tokens (`<pad>`, `<unk>`)
- **GPU Support**: Automatic CUDA detection for accelerated training
- **Standalone Inference**: Easy-to-use prediction script for new text samples

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch pandas numpy matplotlib
```

## Model Architecture

The `Text_Classifier` model includes:
- **Embedding Layer**: Converts word indices to dense vectors
- **RNN Layer**: LSTM/GRU/RNN with configurable layers and bidirectionality
- **Fully Connected Layer**: Maps hidden states to sentiment classes

### Hyperparameters
- Embedding Dimension: 64
- Hidden Dimension: 128
- Number of Layers: 3
- Bidirectional: True
- Sequence Length: 100
- Number of Classes: 3 (Negative, Neutral, Positive)

## Workflow

### 1. Data Preprocessing ([PreProcessing.ipynb](PreProcessing.ipynb))

Prepares the raw data for training:

- **Load Dataset**: Reads the CSV file containing text and sentiment labels
- **Data Cleaning**: 
  - Removes missing values and unnecessary columns
  - Converts text to lowercase
  - Removes URLs, mentions (@username), and hashtags
  - Tokenizes text into word lists
- **Vocabulary Building**:
  - Creates vocabulary from all unique words
  - Adds special tokens: `<pad>` (index 0) and `<unk>` (index 1)
- **Save Artifacts**: Exports vocabulary to `vocab.pkl` for inference

### 2. Model Training ([SentimentAnaylsis_LSTM.ipynb](SentimentAnaylsis_LSTM.ipynb))

Complete training pipeline:

- **Data Loading**: Loads preprocessed data and vocabulary
- **Model Initialization**: Creates the LSTM classifier with specified architecture
- **Training Loop**:
  - Batch processing with DataLoader
  - CrossEntropyLoss for multi-class classification
  - Adam optimizer with learning rate scheduling
  - Gradient clipping for stable training
- **Evaluation**: Tests model performance on validation set
- **Model Saving**: Exports trained weights to `sentiment_model.pkl`

### 3. Inference ([Sentiment_LSTM.py](Sentiment_LSTM.py))

Run predictions on new text:

```python
python Sentiment_LSTM.py
```

The script:
1. Loads the saved vocabulary and model weights
2. Preprocesses input text using the same cleaning pipeline
3. Converts text to indices and pads/truncates to sequence length
4. Runs inference and returns sentiment predictions

Example output:
```
--- Predictions ---
Sentence: I absolutely love this product, it is amazing!
Prediction: 2 (Positive)
--------------------
Sentence: The service was terrible and the food was cold.
Prediction: 0 (Negative)
--------------------
```

## Usage

### Training a New Model

1. Prepare your dataset with two columns: `Comment` (text) and `Sentiment` (label)
2. Run [PreProcessing.ipynb](PreProcessing.ipynb) to clean data and build vocabulary
3. Open [SentimentAnaylsis_LSTM.ipynb](SentimentAnaylsis_LSTM.ipynb) and execute all cells
4. Monitor training progress and validation metrics
5. Model weights will be saved as `sentiment_model.pkl`

### Making Predictions

```python
from Sentiment_LSTM import predict_sentence, Text_Classifier
import torch
import pickle

# Load vocabulary
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Initialize and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Text_Classifier(len(vocab), 64, 128, 3, 3, "LSTM", True).to(device)
model.load_state_dict(torch.load('sentiment_model.pkl', map_location=device))

# Predict sentiment
text = "This is an excellent product!"
prediction = predict_sentence(text, model, vocab, device)
print(f"Sentiment: {prediction}")  # 0=Negative, 1=Neutral, 2=Positive
```

## Text Preprocessing Pipeline

The `clean_text()` function performs:
1. Convert to lowercase
2. Remove URLs (http://, https://, www.)
3. Remove mentions (@username)
4. Remove hashtags (#)
5. Split into word tokens

## Model Configurations

You can experiment with different architectures by modifying:

```python
model = Text_Classifier(
    vocab_size=len(vocab),
    embed_dim=64,           # Embedding dimension
    hidden_dim=128,         # LSTM hidden size
    num_layers=3,           # Number of LSTM layers
    number_classes=3,       # Sentiment classes
    model_type="LSTM",      # Options: "LSTM", "GRU", "RNNs"
    bidirectional=True      # Use bidirectional processing
)
```

Modify this file to use different vocabularies or sample datasets.

## Model Parameters

- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: CrossEntropyLoss
- **Default Epochs**: 5
- **Default Embedding Dimension**: 10

## Key Features

- **Deterministic Label Mapping**: Labels are sorted before mapping to ensure reproducibility
- **Special Token Handling**: 
  - `<pad>` (index 0): Padding token
  - `<unk>` (index 1): Unknown words token
- **Saved Artifacts**: Label mapping saved as JSON for inference
- **Flexible Architecture**: Easy to modify vocabulary and embedding dimensions

## Output Files

After running the preprocessing notebook:
- `dataset/cleaned_sentiment_data.csv`: Cleaned dataset without missing values
- `dataset/label_mapping.json`: Sentiment label to numeric mapping

## Notes

- Ensure the dataset folder exists before running the notebook
- The preprocessing notebook uses GPU if available (CUDA detection included)
- Text cleaning removes URLs, mentions, and hashtags automatically
- Labels are automatically mapped to numeric format (0, 1, 2, etc.)

## Future Improvements

- [ ] Add data augmentation
- [ ] Implement early stopping
- [ ] Add validation set
- [ ] Support for pre-trained embeddings (Word2Vec, GloVe)
- [ ] Hyperparameter tuning
- [ ] Model checkpointing
- [ ] Batch processing for large datasets
- [ ] Cross-validation
