# MiniGPT

MiniGPT is a minimal, educational implementation of a GPT-style decoder-only Transformer trained at the character level using PyTorch.
The project is designed to be easy to read, modify, and experiment with, making it ideal for learning how Transformers work end-to-end.

##  Project Structure

```
MiniGPT/
│
├── model.py              # Main training + generation script
├── Mahabharata.txt       # Training corpus
├── gpt-dev.ipynb         # Development / experimentation notebook
├── matrix-multi.ipynb    # Attention & matrix operation experiments
├── requirements.txt      # Python dependencies
└── README.md
```

## Main Script

`model.py`: model.py is the main entry point of the project.

- Builds a GPT-style Transformer (multi-head self-attention + feed-forward blocks)
- Trains the model using next-token prediction
- Periodically evaluates train/validation loss
- Generates text samples after training

## How to Train
1. Install dependencies
```pip install -r requirements.txt```

2. Run training
```python model.py```

The script will:
- Load Mahabharata.txt
- Train the model
- Print training and validation loss
- Generate sample text at the end

If CUDA is available, training will automatically run on GPU.

## Model Details

- Builds a **character-level vocabulary** from `Mahabharata.txt` and creates integer mappings (`stoi`, `itos`) for encoding and decoding text.
- Implements a **decoder-only Transformer language model** (`BiGramLanguageModel`) that:
  1. Uses a **token embedding table** (`nn.Embedding(vocab_size, n_embd)`) to map characters to dense vectors.
  2. Uses a **learned position embedding table** (`nn.Embedding(block_size, n_embd)`) to encode token positions within the context window.
  3. Combines **token and position embeddings** and processes them through a stack of Transformer blocks with **masked multi-head self-attention**, **feed-forward layers**, **residual connections**, and **LayerNorm**.
  4. Projects the final hidden states to **vocabulary logits** using a linear output head (`nn.Linear(n_embd, vocab_size)`).
  5. Computes **cross-entropy loss** during training when target tokens are provided.
  6. Supports **autoregressive text generation** via multinomial sampling in `generate()`, predicting one token at a time using the last context window.
- Trains the model using the **AdamW optimizer** with **cross-entropy loss** on randomly sampled mini-batches from the training split.
- Periodically evaluates **training and validation loss** to monitor convergence.
- Automatically uses **GPU (`cuda`) when available**, otherwise falls back to **CPU** via device auto-detection.

**Hyperparameters (defaults in `model.py`)**
- **split_percent**: 0.8 — train/validation split
- **batch_size**: 32 — number of sequences processed in parallel
- **block_size**: 8 — maximum context length
- **max_iters**: 5000 — total training iterations
- **eval_interval**: 300 — evaluation frequency
- **learning_rate**: 1e-3 — AdamW learning rate
- **eval_iters**: 200 — batches used for evaluation
- **n_embd**: 32 — embedding dimensionality
- **n_head**: 6 — number of attention heads
- **n_layer**: 6 — number of Transformer blocks
- **dropout**: 0.2 — dropout probability (training only)

## Notes

- n_embd must be divisible by n_head
- This project is intended for learning and experimentation
- Easy to extend with:
	1. temperature / top-k sampling
	2. FlashAttention
	3. fused QKV projections
	4. better tokenization (BPE / SentencePiece)

## Attribution

This repository is inspired by **Andrej Karpathy’s excellent video on building GPT from scratch**:  
https://www.youtube.com/watch?v=kCc8FmEb1nY

