# Implementation of FastSpeech 2-Based Text-to-Speech Synthesis

This repository implements and evaluates the FastSpeech 2 Text-to-Speech (TTS) architecture separately for Indian English and Telugu, in accordance with the assignment specifications.

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

---

## Datasets
- **English**: AI4Bharat Svarah crowdsourced corpus (4.93 h train across 3,288 utterances, 1.0 h validation across 668 utterances; 2-class female/male gender conditioning).
- **Telugu**: IIT Madras IndicTTS studio recordings (4.89 h train across 2,509 utterances, 0.98 h validation across 503 utterances; single-speaker female acoustic register, mean $F_0 = 190.2$ Hz).

---

## Setup & Dependencies

Python 3.11 with PyTorch 2.6+ (CUDA 12.x):
```bash
# Clone the repository
git clone https://github.com/pranav-suresh13/Fastspeech2.git
cd Fastspeech2

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Model Checkpoints (80,000 Steps)

Due to GitHub file size limits, the trained checkpoints (step 80,000) for both English and Telugu are hosted externally:
- **Google Drive Checkpoints:** [Download from Google Drive](https://drive.google.com/drive/folders/1MrqNBMvWXgbAy0rG-SVVY9vMP_X4e4Fd?usp=sharing)
- **Kaggle Checkpoint Mirror:** [https://www.kaggle.com/datasets/pranavsuresh13/fastspeech2-checkpoints-80k](https://www.kaggle.com/datasets/pranavsuresh13/fastspeech2-checkpoints-80k)

Place the downloaded checkpoints into their respective folders:
- English checkpoint: `English/checkpoints/80000.pth.tar`
- Telugu checkpoint: `Telugu/checkpoints/80000.pth.tar`

Pre-trained HiFi-GAN vocoder weights (`generator_universal.pth.tar` or `generator_LJSpeech.pth.tar`) should also be placed in `English/hifigan/` and `Telugu/hifigan/`.

---

## Reproducibility & Commands

### 1. Evaluation Inference (Diagnostic Test Sentences E1–E4, T1–T4)
To synthesize all benchmark test sentences and automatically generate the physical-unit variance adaptor plots ($F_0$ in Hz, Energy in dB, phoneme durations):

```bash
# Run English test suite (Female Voice: speaker_id 0, Male Voice: speaker_id 1)
cd English
python inference.py --test_set --restore_step 80000 --speaker_id 0
cd ..

# Run Telugu test suite (Studio Voice: speaker_id 0)
cd Telugu
python inference.py --test_set --restore_step 80000 --speaker_id 0
cd ..
```
Outputs are written to:
- Audios: `Results/{Language}/test_generated_audios/{ID}.wav`
- Plots: `Results/{Language}/variance_adaptor_plots/{duration,pitch,energy,stacked}/{ID}.png`

### 2. Single-Sentence Synthesis
```bash
# English Custom Text
cd English
python inference.py --text "Text to speech synthesis with FastSpeech 2." --restore_step 80000 --speaker_id 0

# Telugu Custom Text
cd Telugu
python inference.py --text "ఈ సమావేశం సమయానికి ప్రారంభమవుతుంది." --restore_step 80000 --speaker_id 0
```

### 3. Preprocessing & Feature Extraction
To re-run dataset preprocessing (mel-spectrogram extraction, pitch computation, energy calculation, and stats generation):
```bash
# English
cd English
python preprocess.py config.yaml

# Telugu
cd Telugu
python preprocess.py config.yaml
```

### 4. Training from Scratch
To launch distributed dual-GPU training (or single-GPU):
```bash
# English
cd English
python train.py -p config.yaml -m config.yaml -t config.yaml

# Telugu
cd Telugu
python train.py -p config.yaml -m config.yaml -t config.yaml
```

---

## Technical Report

A detailed 45-page scientific analysis is provided in [Report/Technical_Report.pdf](Report/Technical_Report.pdf), covering:
1. Complete architectural specifications and base repository modifications.
2. Dataset selection, curation, and forced alignment methodology (MFA).
3. Convergence behavior, loss curves, and analysis of validation loss trajectory.
4. Per-sentence perceptual analysis and phoneme-level variance adaptor dynamics.
5. Comparative evaluation between English crowdsourced multi-speaker and Telugu clean studio data.
