import os
import time
import math
import pickle
import tiktoken  # for GPT-2 encoding

import numpy as np
import torch

from model import GPTConfig, GPT  # assumes your GPT model implementation is in model.py

# ----------------------------
# Configuration
# ----------------------------
out_dir = 'out'
data_dir = os.path.join('data', 'train-en-vi')
batch_size = 12
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float16' if device != 'cpu' else 'float32'
ptdtype = torch.float16 if dtype == 'float16' else torch.float32

# Sampling settings:
sample_interval = 2000  # run sampling every 2000 iterations
num_samples = 1         # number of samples to generate at each sample interval
max_new_tokens = 200    # max tokens to generate
temperature = 0.8
top_k = 200
prompt = "\n"          # prompt to seed the generation

os.makedirs(out_dir, exist_ok=True)

# ----------------------------
# Data loader
# ----------------------------
def get_batch(split):
    # Load data from a memmapped binary file
    bin_path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# ----------------------------
# Determine vocab_size
# ----------------------------
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"Found vocab_size = {vocab_size}")
else:
    vocab_size = 50304

# ----------------------------
# Model Initialization
# ----------------------------
print("Initializing a new GPT model from scratch...")
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf).to(device)

# ----------------------------
# Optimizer & Mixed Precision Setup
# ----------------------------
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
ctx = torch.amp.autocast if device != 'cpu' else lambda **kwargs: torch.no_grad()

# ----------------------------
# Learning Rate Schedule
# ----------------------------
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# Setup Encoding/Decoding with tiktoken
# -----------------------------------------------------------------------------
# Use GPT-2 encoding as default
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# ----------------------------
# Training Loop
# ----------------------------
iter_num = 0
t0 = time.time()

while iter_num < max_iters:
    # Adjust learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Get batch and forward pass
    X, Y = get_batch('train')
    with ctx(device_type=device.split(':')[0], dtype=ptdtype):
        logits, loss = model(X, Y)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    if grad_clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Simple logging
    if iter_num % 100 == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"iter {iter_num}: loss {loss.item():.4f}, lr {lr:.6f}, time {dt*1000:.2f}ms")

    # Run sampling every 'sample_interval' iterations
    if iter_num % sample_interval == 0:
        # Encode the prompt
        start_ids = encode(prompt)
        x_sample = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        # Generate sample(s)
        model.eval()  # set to eval mode for generation
        with torch.no_grad():
            with ctx(device_type=device.split(':')[0], dtype=ptdtype):
                for _ in range(num_samples):
                    y = model.generate(x_sample, max_new_tokens,
                                       temperature=temperature,
                                       top_k=top_k)
                    print("=== Sample Generation ===")
                    print(decode(y[0].tolist()))
                    print("-------------------------")
        model.train()  # revert back to training mode

    iter_num += 1

# ----------------------------
# Save final model checkpoint
# ----------------------------
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'iter_num': iter_num,
    'model_args': model_args,
}
torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
print("Training complete. Checkpoint saved.")
