import torch
import torch.nn as nn
import re
import pickle
import numpy as np

# ==========================================
# 1. RE-DEFINE THE ARCHITECTURE (Copied from your notebook)
# ==========================================
class Text_Classifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, number_classes, model_type="LSTM", bidirectional=False):
        super(Text_Classifier, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        if model_type == "RNNs":
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        elif model_type == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)

        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, number_classes)
        else:
            self.fc = nn.Linear(hidden_dim, number_classes)

    def forward(self, x):
        x = self.embeddings(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ==========================================
# 2. HELPER FUNCTIONS (Copied from your notebook)
# ==========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    words = text.split()
    return words

def text_to_indices(text, vocab):
    # Use 1 for <unk> if word not found (based on your notebook logic where <unk>=1)
    return [vocab.get(word, vocab.get("<unk>", 1)) for word in text]

def predict_sentence(sentence, model, vocab, device, seq_length=100):
    model.eval()
    
    # Preprocess
    cleaned_words = clean_text(sentence)
    indexed_text = text_to_indices(cleaned_words, vocab)
    
    # Pad or Truncate
    if len(indexed_text) > seq_length:
        processed_text = indexed_text[:seq_length]
    else:
        # Pad with 0 (<pad>)
        processed_text = indexed_text + [0] * (seq_length - len(indexed_text))
    
    # Convert to Tensor
    input_tensor = torch.tensor(processed_text, dtype=torch.long).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
    return predicted_class

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Settings (Must match your training Config!)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    NUM_LAYERS = 3
    BIDIRECTIONAL = True
    MODEL_PATH = 'sentiment_model.pkl' # The file you uploaded
    
    # Load Vocabulary
    try:
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        print("✅ Vocabulary loaded.")
    except FileNotFoundError:
        print("❌ Error: 'vocab.pkl' not found. You must save the vocab from the notebook first!")
        exit()

    # Initialize Model
    # Note: We need len(vocab) and number of classes (3)
    model = Text_Classifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        number_classes=3, 
        model_type="LSTM",
        bidirectional=BIDIRECTIONAL
    ).to(DEVICE)

    # Load Weights
    try:
        # map_location ensures it loads on CPU if you don't have a GPU right now
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Model weights loaded successfully.\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        exit()

    # Test Samples
    samples = [
        "I absolutely love this product, it is amazing!",
        "The service was terrible and the food was cold.",
        "It was okay, nothing special.",
        "I hate this, worst experience ever."
    ]

    # Mapping based on your notebook: 0=Negative (?), 1=Neutral (?), 2=Positive (?)
    # (Double check your notebook's label_mapping to be sure!)
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    # Note: If your mapping was different in the notebook (e.g. 0=Neutral), update this dict.
    # Based on your notebook cells, it seems: 0->1(Neutral?), 1->0(Neg?), 2->2(Pos?)
    # You should load 'label_mapping.pkl' if you saved it to be 100% sure.

    print("--- Predictions ---")
    for s in samples:
        pred_idx = predict_sentence(s, model, vocab, DEVICE)
        # Use the label map if you have it, otherwise just print the number
        print(f"Sentence: {s}")
        print(f"Prediction: {pred_idx} ({sentiment_map.get(pred_idx, 'Unknown')})")
        print("-" * 20)