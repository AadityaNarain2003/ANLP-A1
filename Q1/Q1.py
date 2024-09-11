#This code loads the data and creates 3 separate sets of test,train and validate
'''
The use of this piece of code is just to get the right test train and validate split
'''
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('../Data/6_worded_2.csv', header=None)

train_data, temp_data = train_test_split(data, test_size=0.30, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.33, random_state=42) 

print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")

'''
I now need to create a vocabulary of the words in the output files's train data
This enables me to create a word vocabulary
'''
word_to_index = {}
index = 1
word_to_index['<UNK>'] = 0
for _, row in train_data.iterrows():
    for word in row:
        if word not in word_to_index:
            word_to_index[word] = index
            index += 1



index_to_word = [''] * (len(word_to_index))
for word, idx in word_to_index.items():
    index_to_word[idx] = word
    
'''
This enables me to load the glove 6B 300 dimension embedding into the glove embedding matrix
'''
import numpy as np

glove_file = '../Data/glove.6B.300d.txt' 

embedding_dim = 300

glove_embeddings = {}
with open(glove_file, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_embeddings[word] = vector

vocab_size = len(word_to_index)
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, idx in word_to_index.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[idx] = embedding_vector
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
embedding_matrix[word_to_index['<UNK>']] = np.random.normal(scale=0.6, size=(embedding_dim,))
print(embedding_matrix)



#Now I need to create the model
import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, vocab_size,hidden_state,p):
        super(CustomModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=5 * 300, out_features=hidden_state),
            nn.ReLU(),
            nn.Dropout(p=p),  # Add dropout layer
            nn.Linear(in_features=hidden_state, out_features=vocab_size)
        )
    def forward(self, x):
        # Forward pass through the network
        # Softmax is typically applied during loss calculation, so it's not included here
        x=self.model(x)
        return x


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math

#prepare data in tensor format 
def prepare_data(data, word_to_index, seq_length=5):
    X = []
    y = []
    for _, row in data.iterrows():
        sequence = [word_to_index.get(word, 0) for word in row[:seq_length]]
        target = word_to_index.get(row[seq_length], 0)
        X.append(sequence)
        y.append(target)
    return torch.LongTensor(X), torch.LongTensor(y)

X_train, y_train = prepare_data(train_data, word_to_index)
X_val, y_val = prepare_data(val_data, word_to_index)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize the model, loss function, and optimizer


lrs = [0.01, 0.001, 0.0001]
epochs = [10,3, 5]
hidden_dims = [300,100,300,500]
p=[0.5,0.1,0.5,0.7]

for l in lrs[0:1]:
    for num_epochs in epochs[0:1]:
        for hd in hidden_dims[0:1]:
            for x in p[0:1]:
                print(l,num_epochs,hd,x)
                model = CustomModel(vocab_size,hd,x)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=l, weight_decay=1e-5 )

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # Convert embedding_matrix to a torch tensor
                embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32).to(device)

                model.to(device)

                for epoch in range(num_epochs):
                    model.train()
                    total_loss = 0
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                        # Convert word indices to embeddings
                        batch_embeddings = embedding_matrix[batch_X].view(batch_X.size(0), -1)

                        optimizer.zero_grad()
                        outputs = model(batch_embeddings)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()

                    # Average the training loss
                    avg_train_loss = total_loss / len(train_loader)
                    # Validation
                    model.eval()
                    val_loss = 0
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                            batch_embeddings = embedding_matrix[batch_X].view(batch_X.size(0), -1)
                            outputs = model(batch_embeddings)
                            loss = criterion(outputs, batch_y)
                            val_loss += loss.item()

                            _, predicted = outputs.max(1)
                            total += batch_y.size(0)
                            correct += predicted.eq(batch_y).sum().item()

                    avg_val_loss = val_loss / len(val_loader)  # Correctly average   the validation loss
                    perplexity = math.exp(avg_val_loss)  # Use validation loss for perplexity

                    print(f"Epoch {epoch + 1}/{num_epochs}")
                    print(f"Training Loss: {avg_train_loss:.4f}")
                    print(f"Validation Loss: {avg_val_loss:.4f}")
                    print(f"Perplexity: {perplexity:.2f}")

                # Save the model
                torch.save(model.state_dict(), 'trained_model.pth')

                import torch
                import torch.nn as nn
                from torch.utils.data import DataLoader, TensorDataset

                # Assuming you have these from your training script:
                # CustomModel, word_to_index, index_to_word, embedding_matrix, test_data

                def prepare_data(data, word_to_index, seq_length=5):
                    X = []
                    y = []
                    for _, row in data.iterrows():
                        sequence = [word_to_index.get(word, 0) for word in row[:seq_length]]
                        target = word_to_index.get(row[seq_length], 0)
                        X.append(sequence)
                        y.append(target)
                    return torch.LongTensor(X), torch.LongTensor(y)

                def load_model(model_path, vocab_size):
                    model = CustomModel(vocab_size)
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    return model

                # Prepare test data
                X_test, y_test = prepare_data(test_data, word_to_index)
                test_dataset = TensorDataset(X_test, y_test)
                test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

                ## Load the trained model
                #model_path = 'trained_model.pth'
                #vocab_size = len(word_to_index)
                #model = load_model(model_path, vocab_size)
#
                ## Convert embedding_matrix to a torch tensor
                #embedding_matrix = torch.FloatTensor(embedding_matrix)
#
                ## Set up the device
                #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                #model.to(device)
#
                ## Evaluation
                #model.eval()
                #criterion = nn.CrossEntropyLoss()
                #test_loss = 0
                #correct = 0
                #total = 0
#
                #with torch.no_grad():
                #    for batch_X, batch_y in test_loader:
                #        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                #        batch_embeddings = embedding_matrix[batch_X].view(batch_X.size(0), -1)
#
                #        outputs = model(batch_embeddings)
                #        loss = criterion(outputs, batch_y)
                #        test_loss += loss.item()
#
                #        _, predicted = outputs.max(1)
                #        total += batch_y.size(0)
                #        correct += predicted.eq(batch_y).sum().item()
#
                ## Calculate and print metrics
                #avg_loss = test_loss / len(test_loader)
                #accuracy = 100. * correct / total
                #perplexity = math.exp(avg_loss)  # Use validation loss for perplexity
#
                #print(f"Test Loss: {avg_loss:.4f}")
                #print(f"Test Perplexity: {perplexity:.4f}")

                ### Dropout, Dimensions, Optimizer, Learning Rate,

                optims = ["sgd", "adam"]