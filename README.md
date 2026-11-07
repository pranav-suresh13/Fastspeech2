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
Refer to `requirements.txt` for environment requirements. Install via:
```bash
pip install -r requirements.txt
```

## Model Checkpoints (80,000 Steps)
Due to GitHub file size limits, the trained checkpoints (step 80,000) for both English and Telugu are hosted on Google Drive:
- **Google Drive Checkpoints:** [https://drive.google.com/drive/folders/1MrqNBMvWXgbAy0rG-SVVY9vMP_X4e4Fd?usp=sharing](https://drive.google.com/drive/folders/1MrqNBMvWXgbAy0rG-SVVY9vMP_X4e4Fd?usp=sharing)
- **Kaggle Mirror:** [https://www.kaggle.com/datasets/pranavsuresh13/fastspeech2-checkpoints-80k](https://www.kaggle.com/datasets/pranavsuresh13/fastspeech2-checkpoints-80k)

To run inference using these checkpoints, place `80000.pth.tar` into `English/checkpoints/` and `Telugu/checkpoints/` respectively.

## Quick Inference Commands
Run all assignment benchmark test sentences (E1–E4, T1–T4):
```bash
# English Test Sentences (Female Voice / Speaker 0)
cd English
python inference.py --test_set --restore_step 80000 --speaker_id 0

# Telugu Test Sentences (Studio Voice / Speaker 0)
cd Telugu
python inference.py --test_set --restore_step 80000 --speaker_id 0
```
Outputs and full-resolution variance adaptor plots are automatically saved under `Results/English/` and `Results/Telugu/`.

