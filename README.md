# mini_gpt

A small PyTorch-based GPT-like character model for learning and generation.

## Files

- `config.py` — model and training configuration
- `dataset.py` — data loading, vocabulary building, and batching
- `model.py` — GPT model implementation
- `train.py` — training loop
- `generate.py` — text generation script
- `model.pt` — saved model weights (ignored by default for future pushes)

## Setup

1. Create a Python environment

```bash
python -m venv venv
source venv/bin/activate
```

2. Install requirements

```bash
pip install -r requirements.txt
```

3. Configure device

Edit `config.py` and set `DEVICE = "cuda"` if you have GPU support, otherwise keep `DEVICE = "cpu"`.

## Usage

Train the model:

```bash
python train.py
```

Generate text from the trained model:

```bash
python generate.py
```

