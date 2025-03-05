# README.md

This repository provides a cascaded multimodal framework to train a model which evaluates a social communication severity level of children with ASD.
The original paper is submitted to Interspeech 2025.

![Image](https://github.com/user-attachments/assets/1c30dc0b-70cb-46bb-a3f4-5456002fcfa1)

## Requirements
It requires wave files and a csv file which contains wave file path.

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run scripts
#### Decoding
The ASR module decodes the target waveforms.
```bash
python3 main.py --mode decode
```

#### Extract Embedding
The speech and language foundation models extract embeddings from the raw waveform and transcribed text.
```bash
python3 main.py --mode embed
```

#### Training
Train the multimodal assessment model
```bash
python3 main.py --mode train
```

#### Inference
Get the final predicted results with ensembled models
```bash
python3 main.py --mode inference
```

#### Run the whole code
If you want to run the whole code in a row, simply run this code
```bash
python3 main.py --mode all
```