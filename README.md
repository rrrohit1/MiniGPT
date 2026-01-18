# MiniGPT

A tiny character-level bigram language model implemented in PyTorch. This repository provides a compact, educational demo of a token-to-token bigram LM implemented in `model.py`.

**Project**: Shows how to build a minimal autoregressive character model, train it on `Mahabharata.txt`, and generate text.

**Files**
- **`model.py`**: training + generation script (single-file demo). Implements a small `BiGramLanguageModel` that uses token embeddings, positional embeddings, and a linear head to produce logits over the vocabulary.
- **[gpt-dev.ipynb](gpt-dev.ipynb)**: notebook with step-by-step cells for downloading data, tokenization, training, a simpler embedding-only demo, and a short self-attention exploration.
- **`Mahabharata.txt`**: training text (should be in the repo root).

**Requirements**
- Python 3.8+ (3.10+ recommended)
- PyTorch (CPU or CUDA build depending on your hardware)

Install PyTorch if needed, for example:

```
pip install torch
```

**Quick start**

1. Ensure [Mahabharata.txt](Mahabharata.txt) is present (the notebook includes a wget command to download it).
2. Run the demo:

```
python model.py
```

The script trains for a modest number of iterations, prints train/val losses periodically, and finally generates 500 characters starting from a zero context.

**What `model.py` does**
- Builds a character vocabulary from [Mahabharata.txt](Mahabharata.txt).
- Implements `BiGramLanguageModel` which:
	- Uses a token embedding table (`nn.Embedding(vocab_size, n_embd)`) and a position embedding table (`nn.Embedding(block_size, n_embd)`).
	- Adds token + position embeddings and projects to vocabulary logits with a linear head (`nn.Linear(n_embd, vocab_size)`).
	- Computes cross-entropy training loss when targets are provided and supports autoregressive sampling via multinomial sampling in `generate()`.
- Trains with `AdamW` and cross-entropy loss using randomly sampled batches.
- Automatically uses `cuda` when available (`device` auto-detection).

**Notebook**
- [gpt-dev.ipynb](gpt-dev.ipynb) walks through data download, tokenization, batching, building a minimal bigram model (earlier/simple variant), training loops, and a brief self-attention demonstration.

**Hyperparameters (defaults in `model.py`)**
- **`split_percent`**: 0.8 (train/val split)
- **`batch_size`**: 32
- **`block_size`**: 8
- **`max_iters`**: 3000
- **`eval_interval`**: 300
- **`learning_rate`**: 1e-2
- **`eval_iters`**: 200
- **`n_embd`**: 32 (embedding dimensionality)

**Notes & tips**
- The model uses a small embedding dimension and context (`block_size`) so it trains quickly as a pedagogical example.
- Generation uses multinomial sampling from the softmaxed logits; you can customize sampling in the notebook or modify `model.py` if you want greedy, temperature, or top-k/top-p sampling.
- If you want to run longer training or larger models, adjust the hyperparameters at the top of `model.py`.

If you'd like, I can also:
- Run a quick training run and show output logs.
- Add a small note explaining how to change sampling (temperature/top-k) in `model.py`.
- Commit the README update and create a short changelog entry.
