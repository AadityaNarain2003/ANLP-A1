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

# Function to pad sequences and create necessary masks for Transformer training
def collate_fn(batch):
    # Extract input sequences, target sequences, and their lengths from the batch
    sentences = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    lengths = [item[2] for item in batch]

    # Convert lists of sequences to tensors
    sentences = list(map(lambda s: torch.tensor(s, dtype=torch.float32), sentences))
    targets = list(map(lambda t: torch.tensor(t, dtype=torch.long), targets))

    # Sort sequences by length for efficient batch processing
    lengths, sorted_indices = torch.sort(torch.tensor(lengths), descending=True)
    sen = []
    tar = []
    for i in sorted_indices:
        sen.append(sentences[i])
        tar.append(targets[i])

    # Set padding values for sentences and targets
    padding_value_sentences = 0
    padding_value_targets = word_to_index['<pad>']

    # Pad the sequences to the maximum length in the batch
    padded_sentences = pad_sequence(sen, batch_first=True, padding_value=padding_value_sentences)
    padded_targets = pad_sequence(tar, batch_first=True, padding_value=padding_value_targets)

    # Create a mask to ignore padding tokens during loss computation
    pad_token_index = word_to_index['<pad>']
    tgt_key_padding_mask = (padded_targets == pad_token_index).type(torch.FloatTensor)
    
    return padded_sentences, padded_targets, lengths, tgt_key_padding_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        
        # Create a long enough P matrix
        self.encoding = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(positions * div_term)
        self.encoding[:, 1::2] = torch.cos(positions * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        # x is of shape (batch_size, seq_len, d_model)
        return x + self.encoding[:, :x.size(1)]


class WordTransformerDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, nhead=8, max_len=512):
        super().__init__()
        self.hidden_dim = hidden_size  # Set the hidden size for the model

        # Linear layer to project input embeddings to the hidden size
        self.embedding_layer = nn.Linear(input_size, hidden_size)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=hidden_size, max_len=max_len)
        
        # Define a single Transformer Decoder layer
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,  # Set the hidden size for the decoder
            nhead=nhead,  # Number of attention heads
            batch_first=True  # Ensure batch dimension is first
        )

        # Stack multiple Transformer Decoder layers
        self.transformer = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer, 
            num_layers=num_layers  # Number of decoder layers
        )

        # Linear layer to project the decoder output to the vocabulary size
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, tgt_mask=None, tgt_key_padding_mask=None):
        # Pass the inputs through the embedding layer
        embedded_inputs = self.embedding_layer(inputs)
        
        # Add positional encoding to the embeddings
        embedded_inputs = self.positional_encoding(embedded_inputs)
        
        # Initialize the memory (context) for the Transformer as zeros
        batch_dim, seq_len, _ = embedded_inputs.shape
        init_memory = torch.zeros(batch_dim, seq_len, self.hidden_dim, device=embedded_inputs.device)

        # Pass the embedded inputs through the Transformer Decoder
        transformer_output = self.transformer(
            tgt=embedded_inputs, 
            memory=init_memory, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project the Transformer output to the vocabulary size
        final_output = self.output_layer(transformer_output)
        return final_output
    # Function to create a mask to prevent the model from attending to future tokens during training
    def generate_square_subsequent_mask(self, size):
        subsequent_mask = torch.triu(torch.ones(size, size, dtype=torch.float32))
        subsequent_mask = subsequent_mask.transpose(0, 1)
        subsequent_mask = subsequent_mask.masked_fill(subsequent_mask == 0, float('-inf'))
        subsequent_mask = subsequent_mask.masked_fill(subsequent_mask == 1, float(0.0))
        return subsequent_mask

# Function to train the Transformer Decoder model
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        
        # Iterate over the training data in mini-batches
        for batch_idx, (input_seqs, target_seqs, seq_lengths, tgt_key_padding_mask) in enumerate(train_loader):
            optimizer.zero_grad()  # Clear previous gradients

            # Forward pass through the model
            output = model(input_seqs, tgt_key_padding_mask=tgt_key_padding_mask)
            output = output.transpose(1, 2)  # Adjust output for loss calculation

            # Calculate the loss between model predictions and target sequences
            loss = criterion(output, target_seqs)
            loss.backward()  # Backpropagate the error
            optimizer.step()  # Update model parameters

        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            # Iterate over the validation data in mini-batches
            for input_seqs, target_seqs, seq_lengths, tgt_key_padding_mask in val_loader:
                output = model(input_seqs, tgt_key_padding_mask=tgt_key_padding_mask)
                output = output.transpose(1, 2)  # Adjust output for loss calculation
                
                # Accumulate the validation loss
                val_loss += criterion(output, target_seqs).item()

        # Print the training and validation loss for the current epoch
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss/len(val_loader)}')


# Hyperparameters and setup
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.1
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

# Create datasets and data loaders
train_dataset = Sentence_With_Target(train_data, train_targets)
val_dataset = Sentence_With_Target(val_data, val_targets)
test_dataset = Sentence_With_Target(test_data, test_targets)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader=DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Initialize model, criterion, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
model = WordTransformerDecoder(input_size=embedding_dim, hidden_size=256, output_size=len(vocab)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# Training loop
train_loss =0
val_loss = 0

for epoch_num in tqdm(range(NUM_EPOCHS), desc="Epochs"):
    model.train()
    epoch_train_loss = 0.0
    
    for batch_data in train_loader:
        inputs, targets, mask_lengths, tgt_padding_mask = [item.to(device) for item in batch_data]
        tgt_mask = model.generate_square_subsequent_mask(targets.size(1)).to(device)
        
        optimizer.zero_grad()
        predicted_output = model(inputs, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        predicted_output = predicted_output.reshape(-1, len(vocab))
        targets = targets.view(-1)
        
        loss_value = criterion(predicted_output, targets)
        loss_value.backward()
        optimizer.step()
        
        epoch_train_loss += loss_value.item()
    
    train_loss=(epoch_train_loss / len(train_loader))
    
    model.eval()
    epoch_val_loss = 0.0
    
    with torch.no_grad():
        for batch_data in val_loader:
            inputs, targets, mask_lengths, tgt_padding_mask = [item.to(device) for item in batch_data]
            tgt_mask = model.generate_square_subsequent_mask(targets.size(1)).to(device)
            
            predicted_output = model(inputs, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
            predicted_output = predicted_output.reshape(-1, len(vocab))
            targets = targets.view(-1)
            
            loss_value = criterion(predicted_output, targets)
            epoch_val_loss += loss_value.item()
    
    val_loss=(epoch_val_loss / len(val_loader))
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch_num + 1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train Perplexity: {np.exp(train_loss):.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Perplexity: {np.exp(val_loss):.4f}")
    
# Save the model
torch.save(model.state_dict(), 'trained_model.pth')

# Testing Loop
model.eval()
test_loss = 0.0

with torch.no_grad():
    for batch_data in test_loader:
        inputs, targets, mask_lengths, tgt_padding_mask = [item.to(device) for item in batch_data]
        tgt_mask = model.generate_square_subsequent_mask(targets.size(1)).to(device)
        
        predicted_output = model(inputs, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        predicted_output = predicted_output.reshape(-1, len(vocab))
        targets = targets.view(-1)
        
        loss_value = criterion(predicted_output, targets)
        test_loss += loss_value.item()

    test_loss /= len(test_loader)

print(f'Test Loss: {test_loss:.4f}')
