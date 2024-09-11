import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from itertools import chain
from random import shuffle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Function to load and preprocess data from a text file
def load_and_preprocess_data(file_path):
    with open(file_path, 'r') as f:
        # Read each line, remove leading/trailing whitespace, and filter out empty lines
        sentences = [line.strip() for line in f if line.strip()]

    # Shuffle the list of sentences to ensure randomness
    shuffle(sentences)

    # Tokenize each sentence into words
    tokens = [word_tokenize(sentence) for sentence in sentences]
    
    return tokens

# Function to create a vocabulary from tokenized sentences
def create_vocab(tokens):
    # Flatten the list of tokenized sentences to get a list of all words
    all_words = list(chain.from_iterable(tokens))
    
    # Count the frequency of each word and create a vocabulary list
    word_freq = pd.Series(all_words).value_counts()
    vocab = word_freq.index.tolist()
    
    # Add special tokens for unknown words and padding
    vocab.extend(['<unk>', '<pad>'])
    
    # Create a mapping from words to indices and vice versa
    word_to_index = {}
    index_to_word = {}
    for index, word in enumerate(vocab):
        word_to_index[word] = index
    for word, index in word_to_index.items():
        index_to_word[index] = word
    
    return vocab, word_to_index, index_to_word

# Function to load GloVe embeddings from a file
def load_glove_embeddings(glove_file, embedding_dim):
    glove_embeddings = {}
    
    # Read each line of the GloVe file
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Split the line into the word and its corresponding embedding vector
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word] = vector

    return glove_embeddings

# Function to create word embeddings using the loaded GloVe embeddings
def create_word_embeddings(vocab, glove_embeddings, embedding_dim):
    # Create an embedding for unknown words by averaging all GloVe embeddings
    unk_embedding = np.mean(list(glove_embeddings.values()), axis=0)
    
    # Create a zero vector for padding
    pad_embedding = np.zeros(embedding_dim)
    
    # Map each word in the vocabulary to its GloVe embedding
    word2embedding = {word: glove_embeddings.get(word, unk_embedding) for word in vocab}
    
    # Add the padding embedding to the mapping
    word2embedding['<pad>'] = pad_embedding
    
    return word2embedding

# Function to split the data into training, validation, and test sets
def split_data(tokens):
    # Split the data into 70% training and 30% testing
    train_tokens, test_tokens = train_test_split(tokens, test_size=0.3, random_state=42)
    
    # Further split the test set into 20% validation and 10% testing
    val_tokens, test_tokens = train_test_split(test_tokens, test_size=0.33, random_state=42)
    
    return train_tokens, val_tokens, test_tokens

# Function to prepare data by converting words to embeddings and indices
def prepare_data(tokens, word2embedding, word_to_index):
    data = []
    targets = []

    # Iterate over each tokenized sentence
    for sentence in tqdm(tokens, desc="Preparing data"):
        # Convert each word in the sentence to its corresponding embedding
        sentence_embeddings = [
            word2embedding.get(word, word2embedding['<unk>']) for word in sentence
        ]
        data.append(sentence_embeddings)

        # Convert each word in the sentence to its corresponding index
        sentence_indices = [
            word_to_index.get(word, word_to_index['<unk>']) for word in sentence
        ]
        targets.append(sentence_indices)

    return data, targets

# Custom Dataset class to handle input sequences and their corresponding targets
class Sentence_With_Target(Dataset):
    def __init__(self, data, targets):
        self.sentences = data  # Store the input sequences (embeddings)
        self.labels = targets  # Store the target sequences (indices)
    
    def __len__(self):
        # Return the total number of samples
        return len(self.sentences)
    
    def __getitem__(self, index):
        # Get the input sequence and remove the last word (to predict it)
        input_seq = self.sentences[index][:-1]
        
        # Get the target sequence and remove the first word (as it's already known)
        target_seq = self.labels[index][1:]
        
        # Get the length of the input sequence
        seq_length = len(input_seq)
        
        return input_seq, target_seq, seq_length

def collate_fn(batch):
    # Extract input sequences, target sequences, and their lengths from the batch
    sentences = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    lengths = [item[2] for item in batch]

    # Convert lists of sequences to tensors
    sentences = list(map(lambda s: torch.tensor(s, dtype=torch.float32), sentences))
    targets = list(map(lambda t: torch.tensor(t, dtype=torch.long), targets))
    
    padding_value_sentences = vocab_size-1
    padding_value_targets = word_to_index['<pad>']

    # Pad the sequences to the maximum length in the batch
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=padding_value_sentences)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=padding_value_targets)
    
    return padded_sentences, padded_targets, lengths


# Hyperparameters and setup
NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001
embedding_dim = 300
glove_file = '../Data/glove.6B.300d.txt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load data
tokens = load_and_preprocess_data('../Data/test_sentences.txt')
vocab, word_to_index, index_to_word = create_vocab(tokens)
glove_embeddings = load_glove_embeddings(glove_file, embedding_dim)
word2embedding = create_word_embeddings(vocab, glove_embeddings, embedding_dim)
train_tokens, val_tokens, test_tokens = split_data(tokens)
train_data, train_targets = prepare_data(train_tokens, word2embedding, word_to_index)
val_data, val_targets = prepare_data(val_tokens, word2embedding, word_to_index)
test_data, test_targets = prepare_data(test_tokens, word2embedding, word_to_index)
vocab_size=len(vocab)

# Create datasets and data loaders
train_dataset = Sentence_With_Target(train_data, train_targets)
val_dataset = Sentence_With_Target(val_data, val_targets)
test_dataset = Sentence_With_Target(test_data, test_targets)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader=DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

class WordLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(WordLSTM, self).__init__()
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Define the linear layer to map LSTM outputs to the output dimension
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Store the dimensions for initialization
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden_state):
        # Forward pass through the LSTM layer
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        
        # Pass the LSTM output through the fully connected layer
        output = self.fc(lstm_out)
        
        return output, new_hidden_state

    def initialize_hidden_state(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        )
        
# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
print(word_to_index['<pad>'])
print(vocab_size)
# Initialize model, loss function, optimizer, and scheduler
model = WordLSTM(input_dim=embedding_dim, hidden_dim=300, output_dim=vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=(vocab_size-1))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# Training and validation loop
for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    epoch_train_loss = 0
    for batch_idx, (x_batch, y_batch, _) in enumerate(train_loader):
        #print(x_batch.shape)  # Should be [batch_size, max_seq_len, embedding_dim]
        #print(y_batch.shape)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Initialize hidden state
        hidden_state = model.initialize_hidden_state(x_batch.size(0))
        
        optimizer.zero_grad()
        
        # Forward pass
        output_batch, hidden_state = model(x_batch, hidden_state)
        
        # Reshape output and target for loss calculation
        output_batch = output_batch.view(-1, vocab_size)
        y_batch = y_batch.view(-1)
        
        # Calculate loss
        loss = criterion(output_batch, y_batch)
        loss.backward()
        optimizer.step()
        
        # Accumulate training loss
        epoch_train_loss += loss.item()
    
    # Average training loss for this epoch
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_loss = avg_train_loss
    
    # Validation phase
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch, _ in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            hidden_state = model.initialize_hidden_state(x_batch.size(0))
            
            # Forward pass
            output_batch, hidden_state = model(x_batch, hidden_state)
            
            output_batch = output_batch.view(-1, vocab_size)
            y_batch = y_batch.view(-1)
            
            # Calculate validation loss
            loss = criterion(output_batch, y_batch)
            epoch_val_loss += loss.item()
    
    # Average validation loss for this epoch
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_loss = avg_val_loss
    
    # Update learning rate scheduler
    scheduler.step(avg_val_loss)
    
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train Perplexity: {np.exp(train_loss):.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Perplexity: {np.exp(val_loss):.4f}")

# Save the model
torch.save(model.state_dict(), 'trained_model.pth')

# Define the test loop
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Initialize hidden state
            hidden_state = model.initialize_hidden_state(x_batch.size(0))
            
            # Forward pass
            output_batch, hidden_state = model(x_batch, hidden_state)
            
            # Reshape output and target for loss calculation
            output_batch = output_batch.view(-1, vocab_size)
            y_batch = y_batch.view(-1)
            
            # Calculate test loss
            loss = criterion(output_batch, y_batch)
            test_loss += loss.item()
    
    # Average test loss
    avg_test_loss = test_loss / len(test_loader)
    return avg_test_loss

# Initialize test DataLoader
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Test the model
test_loss = test_model(model, test_loader, criterion, device)

print(f"Test Loss: {test_loss:.4f}, Test Perplexity: {np.exp(test_loss):.4f}")
