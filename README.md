# README.md

This repository provides a framework to train a multimodal model to evaluate a social communication severity level of children with ASD.

## How to preprocess (extract embeddings)
It requires a csv file which contains wave file path and transcription.

### Install dependencies
```bash
pip install -r requirements.txt
```
### Run the data preprocessing script
```bash
python extract_embeddings.py
```
## Train and inference
```bash
python main.py --mode train
```

```bash
python main.py --mode inference
```