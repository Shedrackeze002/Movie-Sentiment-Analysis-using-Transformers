# 🎬 Movie Sentiment Analysis using Transformers

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-NLP-yellow.svg)](https://huggingface.co/transformers/)
[![BERT](https://img.shields.io/badge/BERT-Fine--tuned-blue.svg)](https://arxiv.org/abs/1810.04805)

A natural language processing project that performs **binary sentiment classification** on movie reviews using **transformer-based models**. The project fine-tunes BERT and other transformer architectures on the IMDB dataset to classify reviews as positive or negative.

### Deployed live at: https://huggingface.co/spaces/shedrackeze/movie-sentiment-analyzer

## 🎯 Project Objective

Build a sentiment analysis pipeline that:
1. **Preprocesses** movie review text data
2. **Fine-tunes** pre-trained transformer models
3. **Compares** multiple transformer architectures
4. **Achieves** high accuracy on sentiment classification

## 🏗️ Models Implemented

| Model | Description | Architecture |
|-------|-------------|--------------|
| **BERT Fine-tuned** | Full BERT fine-tuning | `bert-base-uncased` + classifier head |
| **DistilBERT** | Distilled BERT classifier | Lightweight transformer |
| **Sentence Transformer** | Sentence embeddings | Sentence-level representations |
| **GPT Demo** | GPT-based classifier | Autoregressive model |

## 📊 Dataset

**IMDB Movie Reviews Dataset**
- **Size:** 50,000 movie reviews
- **Split:** 25,000 training / 25,000 testing
- **Classes:** Binary (Positive / Negative)
- **Average Review Length:** ~230 words

## 🔬 Approach

### 1. Data Preprocessing
- Text cleaning and normalization
- Tokenization using BERT tokenizer
- Padding and truncation to max length
- Creating attention masks

### 2. Model Architecture
```
Input Text → Tokenization → BERT Encoder → Pooled Output → Classifier → Sentiment
```

### 3. Training Strategy
- Learning rate scheduling with warmup
- AdamW optimizer
- Cross-entropy loss
- Early stopping with validation monitoring

## 🛠️ Tech Stack
- **Language:** Python 3.8+
- **Framework:** PyTorch
- **NLP Library:** Hugging Face Transformers
- **Tokenization:** BERT Tokenizer
- **Utilities:** NumPy, Pandas, tqdm

## 📁 Project Structure
```
├── SentimentScope_starter.ipynb  # Main training notebook
├── train/                        # Training data (25K reviews)
│   ├── pos/                      # Positive reviews
│   └── neg/                      # Negative reviews
├── test/                         # Test data (25K reviews)
│   ├── pos/
│   └── neg/
└── artifacts/                    # Saved model weights
    ├── bert_finetuned.pt        # Fine-tuned BERT
    ├── distil_classifier_best.pt
    ├── sentence_head_best.pt
    └── demo_gpt_best.pt
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch transformers pandas numpy tqdm scikit-learn
```

### Running Training
```python
# Open the notebook
jupyter notebook SentimentScope_starter.ipynb

# Or run training script
python train_sentiment.py --model bert --epochs 3 --batch_size 16
```

### Inference Example
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.load_state_dict(torch.load('artifacts/bert_finetuned.pt'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Predict
review = "This movie was absolutely fantastic! Great acting and storyline."
inputs = tokenizer(review, return_tensors='pt', truncation=True, max_length=512)
outputs = model(**inputs)
sentiment = "Positive" if outputs.logits.argmax().item() == 1 else "Negative"
print(f"Sentiment: {sentiment}")
```

## 📈 Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| BERT Fine-tuned | ~92% | ~0.92 |
| DistilBERT | ~90% | ~0.90 |
| Sentence Transformer | ~88% | ~0.88 |

## 🔑 Key Learnings
- **Transfer Learning in NLP:** Leveraging pre-trained language models
- **Fine-tuning Strategies:** Full fine-tuning vs. feature extraction
- **Attention Mechanisms:** Understanding self-attention in transformers
- **Tokenization:** Subword tokenization (WordPiece, BPE)
- **Model Deployment:** Saving and loading PyTorch models

## ⚠️ Large Files (Git LFS)

This repository uses **Git LFS** to track large model files (`.pt` files > 100MB):
- `bert_finetuned.pt` (~440MB)
- `distil_classifier_best.pt`
- `sentence_head_best.pt`
- `demo_gpt_best.pt`

To clone with model weights:
```bash
git lfs install
git clone https://github.com/Shedrackeze002/Movie-Sentiment-Analysis-using-Transformers.git
```

## 📚 References
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)

## 👤 Author
**Eze Nnamdi Shedrack**  
MS in Engineering Artificial Intelligence  
Carnegie Mellon University Africa

---
*This project demonstrates advanced NLP techniques using state-of-the-art transformer models.*
