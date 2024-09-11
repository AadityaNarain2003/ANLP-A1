import csv

# Input and output file paths
input_file = 'Auguste_Maquet_tokens2.csv'
output_file = '6_worded_2.csv'

# Function to create sliding window of 6 words
def create_sliding_window(words, window_size=6):
    windows = []
    for i in range(len(words) - window_size + 1):
        windows.append(words[i:i + window_size])
    return windows

# Read input CSV and write to output CSV with sliding windows
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        # Create sliding windows of 6 words
        sliding_windows = create_sliding_window(row)
        # Write each window as a new row in the output CSV
        for window in sliding_windows:
            writer.writerow(window)
