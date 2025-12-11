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

The preprocessing notebook handles the complete data pipeline:

#### Step 1: Load and Clean Data
- Load raw sentiment data from `dataset/sentiment_data.csv`
- Check dataset shape and identify missing values
- Remove rows with missing comments
- Drop unnecessary columns (e.g., `Unnamed: 0`)
- Save cleaned data to `dataset/cleaned_sentiment_data.csv`

#### Step 2: Text Cleaning
- Convert text to lowercase
- Remove URLs, mentions (@username), and hashtags
- Tokenize sentences into word lists
- Apply cleaning to the 'Comment' column

#### Step 3: Build Vocabulary
- Count word frequencies across all tokenized sentences
- Create word-to-index mapping (starting from index 2)
- Add special tokens:
  - `<pad>` (index 0) for padding sequences
  - `<unk>` (index 1) for unknown words
- Convert text to numerical indices

#### Step 4: Label Processing
- Create deterministic label mapping (sorted unique labels)
- Convert sentiment labels to numeric format (0, 1, 2, etc.)
- Save `label_mapping.json` for reproducibility and inference

#### Step 5: Train-Test Split
- Split dataset: 80% training, 20% testing
- Maintain data distribution

**Key Outputs:**
- `cleaned_sentiment_data.csv`: Preprocessed dataset
- `label_mapping.json`: Sentiment label mappings
- `vocab`: Word-to-index dictionary
- `train_df` and `test_df`: Split datasets ready for model training

### 2. Model Architecture (SentimentAnalysisCNN.py)

The `SentimentAnalysisCNN` class implements a CNN for text classification with:
- Embedding layer for word representations
- Convolutional layers for feature extraction
- Fully connected layers for classification

### 3. Training (Training_loop.py)

Use `Training_loop.py` to train and evaluate the model:

```python
from Training_loop import TrainingLoop, testingLoop
from data import book_sample, word_to_idx, vocab_size, embed_dim
from SentimentAnalysisCNN import SentimentAnalysisCNN
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

## Configuration

### Vocabulary Configuration (data.py)
Edit `data.py` to modify:
- **vocab**: List of words in vocabulary
- **word_to_idx**: Word-to-index mapping
- **embed_dim**: Embedding dimension (default: 10)
- **book_sample**: Training/testing samples

### Model Hyperparameters
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: CrossEntropyLoss
- **Default Epochs**: 5
- **Train-Test Split**: 80/20

## Data Format

### Input Data (sentiment_data.csv)
Expected columns:
- `Comment`: Text data (sentences/reviews)
- `sentiment`: Sentiment labels (string or numeric)

### Preprocessed Data
After running the notebook:
- Tokenized text stored as lists of words
- Numeric indices for each word
- Numeric labels (0, 1, 2, etc.)

## Notes

- **Label Mapping**: Automatically created from sorted unique labels for reproducibility
- **Unknown Words**: Mapped to `<unk>` token (index 1) during inference
- **Padding**: Use `<pad>` token (index 0) for variable-length sequences
- **Deterministic**: Label mapping is saved to ensure consistent predictions
- **Text Column**: Notebook handles both 'Comment' and 'tweet_text' column names

## Future Improvements

- [ ] Add data augmentation
- [ ] Implement early stopping
- [ ] Add validation set
- [ ] Support for pre-trained embeddings (Word2Vec, GloVe)
- [ ] Hyperparameter tuning with grid search
- [ ] Model checkpointing and best model saving
- [ ] Batch processing for large datasets
- [ ] Real-time inference script
- [ ] Visualization of training metrics
- [ ] Cross-validation for robust evaluation
