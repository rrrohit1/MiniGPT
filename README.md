# MiniGPT

A tiny character-level bigram language model implemented in PyTorch. This repository contains a minimal educational example (`model.py`) that trains a simple lookup-based model on `Mahabharata.txt` and demonstrates autoregressive text generation.

**Project**: Demonstrates how a token-to-token embedding can be used as a bigram model to predict the next character, train with a simple loop, and sample generated text.

**Files**
- `model.py`: the training + generation script (single-file demo).
- `Mahabharata.txt`: the training text (expected to be in the same directory).

**Requirements**
- Python 3.8+ (3.10+ recommended)
- PyTorch (CPU or CUDA build depending on your hardware)

Install the torch package if needed, for example:

```
pip install torch
```

**Quick start**

1. Ensure `Mahabharata.txt` is present in the repository root.
2. Run the demo:

```
python model.py
```

The script trains for a small number of iterations, prints train/val losses periodically, and finally generates 500 new characters from a zero context.

**What `model.py` does**
- Builds a character vocabulary from `Mahabharata.txt`.
- Implements `BiGramLanguageModel`, an `nn.Module` with a `nn.Embedding(vocab_size, vocab_size)` table that maps each token to logits for the next token.
- Trains with `AdamW` on cross-entropy loss using randomly sampled batches.
- Selects `cuda` when available, otherwise runs on CPU.

**Hyperparameters (defaults in `model.py`)**
- `split_percent`: 0.8 (train/val split)
- `batch_size`: 32
- `block_size`: 8
- `max_iters`: 3000
- `eval_interval`: 300
- `learning_rate`: 1e-2
- `eval_iters`: 200
