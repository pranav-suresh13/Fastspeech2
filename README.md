# Implementation of FastSpeech 2-Based Text-to-Speech Synthesis

This repository implements and evaluates the FastSpeech 2 Text-to-Speech (TTS) architecture separately for English and Telugu, in accordance with the assignment specifications.

## Project Structure

```text
.
├── English/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Telugu/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/
│   │   └── variance_adaptor_plots/
│   │       ├── duration/
│   │       ├── pitch/
│   │       └── energy/
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/
│       └── variance_adaptor_plots/
│           ├── duration/
│           ├── pitch/
│           └── energy/
├── Report/
│   └── Technical_Report.pdf
├── README.md
└── requirements.txt
```

## Datasets
- **English**: Svarah (~5 hours train, ~1 hour validation)
- **Telugu**: IndicTTS Telugu (~5 hours train, ~1 hour validation)

## Setup & Dependencies
Refer to `requirements.txt` for environment requirements.

