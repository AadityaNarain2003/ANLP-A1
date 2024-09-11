import nltk
import re

# Download the punkt tokenizer for sentence tokenization
nltk.download('punkt')

# Function to clean the sentences
def clean_sentence(sentence):
    # Remove special characters and replace \n with space
    cleaned_sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove special characters
    cleaned_sentence = cleaned_sentence.replace('\n', ' ')  # Replace \n with space
    # Remove multiple spaces if any
    cleaned_sentence = re.sub(' +', ' ', cleaned_sentence).strip()
    return cleaned_sentence

def clean_sentence2(sentence):
    # Replace underscores with spaces
    sentence = sentence.replace('_', ' ')
    
    # Remove special characters
    sentence = re.sub(r'[^\w\s]', '', sentence)
    
    # Replace \n with space and remove extra spaces
    sentence = re.sub(r'\s+', ' ', sentence.replace('\n', ' ')).strip()
    
    return sentence

# Function to check if a sentence has 5 or more words
def is_valid_sentence(sentence):
    words = sentence.split()
    return len(words) > 5

# Function to remove words with numbers followed by 'm' and split the sentences
def remove_and_split(sentence):
    # Regex pattern to find words with numbers followed by 'm'
    pattern = r'\b\d+m\b'
    
    # Find all occurrences of the pattern
    matches = re.findall(pattern, sentence)
    
    if not matches:
        return [sentence]  # Return the original sentence if no match is found
    
    # Remove the words and split the sentence at the positions where the words were removed
    cleaned_sentence = re.sub(pattern, '|', sentence)
    cleaned_sentence = cleaned_sentence.replace('\n', '')
    
    # Split the sentence by the '|' character, which acts as a placeholder for where the word was removed
    fragments = cleaned_sentence.split('|')
    
    # Clean up the fragments by removing leading/trailing spaces and ensuring non-empty fragments
    fragments = [frag.strip() for frag in fragments if frag.strip()]
    
    return fragments

# Load the text file
with open('../Data/Auguste_Maquet.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenize the text into sentences
sentences = nltk.sent_tokenize(text)

# Clean each sentence
cleaned_sentences = [clean_sentence(sentence) for sentence in sentences]

# Filter valid sentences
valid_sentences = [sentence for sentence in cleaned_sentences if is_valid_sentence(sentence)]

# Process each valid sentence to remove unwanted words and split if necessary
processed_sentences = []
for sentence in valid_sentences:
    processed_sentences.extend(remove_and_split(sentence))

# Remove any remaining newline characters from each processed sentence
processed_sentences = [sentence.replace('\n', ' ') for sentence in processed_sentences]

# Write the processed sentences to a text file
with open('Auguste_Maquet_sentences.txt', 'w', encoding='utf-8') as file:
    for sentence in processed_sentences:
        file.write(sentence + '\n')

# Print the first few processed sentences to verify
print(processed_sentences[:3])
