# Sentiment Analysis with CNN

A PyTorch-based sentiment analysis project using Convolutional Neural Networks (CNN) for text classification.

## Project Structure

```
Spamming/
├── dataset/
│   ├── sentiment_data.csv          # Raw dataset
│   ├── cleaned_sentiment_data.csv  # Preprocessed dataset
│   └── label_mapping.json          # Label to index mapping
├── PreProcessing.ipynb             # Data cleaning and preprocessing notebook
├── SentimentAnalysisCNN.py         # CNN model architecture
├── Training_loop.py                # Training and testing functions
├── data.py                         # Vocabulary and sample data
└── README.md                       # This file
```

## Requirements

Install the required packages:

```bash
pip install torch pandas numpy matplotlib
```

## Workflow

### 1. Data Preprocessing (PreProcessing.ipynb)

The preprocessing notebook contains the complete data pipeline:

#### Step 1: Load Raw Data
- Reads `sentiment_data.csv` from the dataset folder
- Displays dataset shape and checks for missing values

#### Step 2: Data Cleaning
- Removes rows with missing values
- Drops unnecessary columns (e.g., 'Unnamed: 0')
- Saves cleaned data to `cleaned_sentiment_data.csv`

#### Step 3: Text Preprocessing
- Converts text to lowercase
- Removes URLs, mentions (@username), and hashtags
- Tokenizes text into word lists
- Applies cleaning to the 'Comment' column

#### Step 4: Vocabulary Building
- Builds vocabulary from all unique words in the dataset
- Adds special tokens: `<pad>` (index 0) and `<unk>` (index 1)
- Converts text to numerical indices using the vocabulary
- Creates deterministic label mapping (sorted by unique sentiment values)
- Saves label mapping to `label_mapping.json` for reproducibility

#### Step 5: Train/Test Split
- Splits dataset into 80% training and 20% testing
- Stores in `train_df` and `test_df` DataFrames

**Run all cells in order to complete the preprocessing pipeline.**

### 2. Model Architecture (SentimentAnalysisCNN.py)

The `SentimentAnalysisCNN` class implements a CNN for text classification with:
- Embedding layer for word representations
- Convolutional layers for feature extraction
- Fully connected layers for classification

### 3. Training (Training_loop.py)

Use the training script to train and evaluate the model:

```python
from Training_loop import TrainingLoop, testingLoop
from SentimentAnalysisCNN import SentimentAnalysisCNN
from data import vocab_size, embed_dim, book_sample
import torch.nn as nn
import torch.optim as optim

# Initialize model
model = SentimentAnalysisCNN(vocab_size, embed_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
TrainingLoop(model, book_sample, criterion, optimizer, epochs=5)

# Test the model
testingLoop(model, book_sample)
```

## Configuration (data.py)

The `data.py` module contains:
- **vocab**: List of words in vocabulary
- **word_to_idx**: Word to index mapping dictionary
- **vocab_size**: Size of vocabulary
- **embed_dim**: Embedding dimension (default: 10)
- **book_sample**: Sample training/testing data

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
