# Named Entity Recognition (NER) with BiLSTM and DistilBERT

This repository contains the implementation of two different architectures for **Named Entity Recognition (NER)**, developed as a final project for the *Machine Learning for Natural Language Processing* course.

## Project Description
The goal of this project is to build and compare two sequence labeling systems to detect entities of type:
- **PER** (Person)
- **ORG** (Organization)
- **LOC** (Location)
- **MISC** (Miscellaneous)

The project follows the **IOB (Inside-Outside-Beginning) tagging scheme** and evaluates performance using entity-level metrics.

## Dataset
The models are trained and tested on the **CoNLL-03** dataset. 
- **Preprocessing**: Tokens were aligned with DistilBERT's sub-word tokenizer using a custom alignment function to ensure correct label mapping.
- **Format**: The input data is provided in JSON format, containing tokens and their corresponding IOB tags.

## Model Architectures

### 1. RNN-based Model
A custom-built Recurrent Neural Network consisting of:
- **Embedding Layer**: Learnable word representations.
- **Bidirectional LSTM (BiLSTM)**: To capture context from both previous and future tokens.
- **TimeDistributed Linear Layer**: A feed-forward layer that maps LSTM outputs to the tag space.

### 2. Fine-tuned BERT Tagger
A Transformer-based approach using:
- **Pre-trained Model**: `distilbert-base-cased`.
- **Classification Head**: A linear layer on top of the hidden states for each token.
- **Fine-tuning**: The model was trained end-to-end to adapt the general language representations to the specific NER task.

## Performance Summary
The models were evaluated using the `seqeval` library. The Transformer-based model significantly outperformed the RNN model, especially in recognizing complex entities like `ORG` and `MISC`.

| Model | Macro F1-Score | Global Accuracy |
| :--- | :---: | :---: |
| **BiLSTM** | ~0.61 | ~92% |
| **DistilBERT** | **~0.87** | **~97%** |

### Key Findings:
- **Context matters**: DistilBERT's attention mechanism handles long-range dependencies much better than the BiLSTM.
- **Label Imbalance**: Both models perform best on `PER` and `LOC`, while `MISC` remains the most challenging category due to its high intra-class variance.

## Setup and Usage

### Requirements
- Python 3.9+
- PyTorch
- Transformers
- seqeval
- Scikit-learn
