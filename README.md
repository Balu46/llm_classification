# Multi-Task AutoQA System using Granite-3B-a800m

This repository implements a modular, hybrid **Dual-Head AutoQA Architecture** for evaluating customer support logs and agent dialogue transcripts. 

To solve "format hallucinations" in generative models, this system jointly outputs a strict classification score (Passed/Failed) and a textual natural language rationale (explaining *why* the conversation passed/failed audit rules).

## 🛠️ Project Structure

```text
llm_classification/
├── config/
│   └── (optional configuration overrides)
├── src/
│   ├── __init__.py
│   ├── config.py         # Model configuration & hyperparameter defaults
│   ├── data.py           # Ingestion, formatting & QA corruption pipeline
│   ├── models.py         # Multi-task architecture & classification head
│   ├── training.py       # Custom train/eval step logic using pure PyTorch
│   └── utils.py          # Save/load checkpoint methods
├── train.py              # Main training script entry point
├── evaluate.py           # Main inference and evaluation entry point
├── requirements.txt      # Project library dependencies
└── README.md             # This instruction manual
```

---

## 📊 Dataset Ingestion & QA Audit Simulation

For representative training data, the pipeline downloads the open-source **[Bitext Customer Support Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)** from Hugging Face. 
* Because public datasets contain exclusively high-quality, correct agent answers, we simulate real-world call center QA audit failures by systematically corrupting 50% of the dialogues using three failure mode heuristics:
  1. **Rude Greeting**: Prepend a hostile agent greeting (e.g. *"What do you want?"*) and blunt close.
  2. **Unhelpful Redirect**: Replace the helpful BPE response with an abrupt department redirection (e.g. *"I can't do anything about that. Call another department."*).
  3. **Abrupt Closing**: Omit any parting confirmation or professional farewell, closing immediately after resolving the query.
* The system automatically generates a structured rationale reflecting the corruption type, matching it to the target binary Pass/Fail label.

---

## ⚡ Quick Start

### 1. Set Up Environment
Activate your virtual environment and install the required dependencies:
```bash
# Activate your local virtual environment
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

### 2. Run Multi-Task Training
Launch the custom multi-task training loop. This loads the model in 4-bit, configures LoRA, runs the epochs, and saves checkpoints in the `./checkpoints/` directory.
```bash
python3 train.py
```

### 3. Evaluate Checkpoints / Inference
Run inference against unseen support logs to check classification accuracy and read the decoded textual rationales:
```bash
python3 evaluate.py
```
