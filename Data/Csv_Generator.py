import csv

# Read the text file
with open('test_sentences.txt', 'r', encoding='utf-8') as file:
    sentences = file.readlines()

# Split each sentence into words and prepare for CSV writing
tokenized_sentences = [sentence.strip().split() for sentence in sentences]

# Write the tokenized sentences to a CSV file
with open('Auguste_Maquet_tokens2.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for tokens in tokenized_sentences:
        writer.writerow(tokens)

# Print the first few tokenized sentences to verify
print(tokenized_sentences[:3])
