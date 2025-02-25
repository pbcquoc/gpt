import os
import requests
import tiktoken
import numpy as np

# Define file paths
base_dir = os.path.dirname(__file__)
input_file = os.path.join(base_dir, 'input.txt')
train_bin = os.path.join(base_dir, 'train.bin')
val_bin = os.path.join(base_dir, 'val.bin')


# Read the dataset
with open(input_file, 'r', encoding='utf-8') as f:
    data = f.read()

# Split data into training and validation sets
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# Encode using tiktoken's GPT-2 encoding
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")

# Save the token ids to binary files
np.array(train_ids, dtype=np.uint16).tofile(train_bin)
np.array(val_ids, dtype=np.uint16).tofile(val_bin)
