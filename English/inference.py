import argparse
import os
import re
from string import punctuation

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from scipy.io import wavfile
from g2p_en import G2p

from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import to_device
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_SENTENCES = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
}


def load_configs(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    preprocess_config = config
    model_config = config.get("model", config)
    train_config = config
    return preprocess_config, model_config, train_config


def read_lexicon(lex_path):
    lexicon = {}
    if os.path.exists(lex_path):
        with open(lex_path, "r", encoding="utf-8") as f:
            for line in f:
                temp = re.split(r"\s+", line.strip("\n"))
                word = temp[0]
                phones = temp[1:]
                if word.lower() not in lexicon:
                    lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return sequence, phones


def plot_variance_features(sample_id, phones_str, duration, pitch, energy, results_dir):
    dur_dir = os.path.join(results_dir, "variance_adaptor_plots", "duration")
    pitch_dir = os.path.join(results_dir, "variance_adaptor_plots", "pitch")
    energy_dir = os.path.join(results_dir, "variance_adaptor_plots", "energy")
    os.makedirs(dur_dir, exist_ok=True)
    os.makedirs(pitch_dir, exist_ok=True)
    os.makedirs(energy_dir, exist_ok=True)

    # 1. Duration plot
    phone_tokens = phones_str.strip("{}").split()
    plt.figure(figsize=(14, 5))
    x = np.arange(len(duration))
    plt.bar(x, duration, color="#4c72b0", edgecolor="black", alpha=0.8)
    if len(phone_tokens) == len(duration) and len(duration) <= 60:
        plt.xticks(x, phone_tokens, rotation=70, fontsize=8)
    plt.title(f"Predicted Duration per Phoneme - [{sample_id}]", fontsize=12, fontweight="bold")
    plt.xlabel("Phoneme Index / Token")
    plt.ylabel("Duration (Frames)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(dur_dir, f"{sample_id}.png"), dpi=200)
    plt.close()

    # 2. Pitch plot
    plt.figure(figsize=(12, 4))
    plt.plot(pitch, color="#c44e52", linewidth=2, label="Predicted Pitch (F0)")
    plt.title(f"Predicted Pitch Contour - [{sample_id}]", fontsize=12, fontweight="bold")
    plt.xlabel("Frame")
    plt.ylabel("Pitch Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(pitch_dir, f"{sample_id}.png"), dpi=200)
    plt.close()

    # 3. Energy plot
    plt.figure(figsize=(12, 4))
    plt.plot(energy, color="#55a868", linewidth=2, label="Predicted Energy")
    plt.title(f"Predicted Energy Contour - [{sample_id}]", fontsize=12, fontweight="bold")
    plt.xlabel("Frame")
    plt.ylabel("Energy Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(energy_dir, f"{sample_id}.png"), dpi=200)
    plt.close()


def synthesize_utterance(model, vocoder, text, sample_id, configs, results_dir, p_control=1.0, e_control=1.0, d_control=1.0):
    preprocess_config, model_config, train_config = configs
    sequence, phones_str = preprocess_english(text, preprocess_config)

    seq_tensor = torch.from_numpy(sequence).long().unsqueeze(0).to(device)
    src_len = torch.tensor([len(sequence)]).long().to(device)
    speakers = torch.tensor([0]).long().to(device)

    with torch.no_grad():
        output = model(
            speakers,
            seq_tensor,
            src_len,
            max(src_len),
            p_control=p_control,
            e_control=e_control,
            d_control=d_control,
        )

        # Output unpacking:
        # output = [output, postnet_output, p_predictions, e_predictions, log_d_predictions, d_rounded, src_masks, mel_masks, src_lens, mel_lens]
        postnet_mel = output[1].transpose(1, 2)
        mel_len = output[9][0].item()
        src_len_val = output[8][0].item()

        duration = output[5][0, :src_len_val].cpu().numpy()
        pitch = output[2][0, :mel_len].cpu().numpy()
        energy = output[3][0, :mel_len].cpu().numpy()

        # Synthesis with vocoder
        wavs = vocoder_infer(postnet_mel, vocoder, model_config, preprocess_config, [mel_len])
        wav = wavs[0]

        # Save audio
        audio_dir = os.path.join(results_dir, "test_generated_audios")
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, f"{sample_id}.wav")
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        wavfile.write(audio_path, sampling_rate, wav)
        print(f"[{sample_id}] Generated audio saved to: {audio_path}")

        # Save variance plots
        plot_variance_features(sample_id, phones_str, duration, pitch, energy, results_dir)
        print(f"[{sample_id}] Variance plots saved under {results_dir}/variance_adaptor_plots/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=None, help="Checkpoint step to restore")
    parser.add_argument("--ckpt", type=str, default=None, help="Direct path to checkpoint .pth.tar")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--test_set", action="store_true", help="Run inference on all official assignment test sentences (E1-E4)")
    parser.add_argument("--text", type=str, default=None, help="Single sentence input")
    parser.add_argument("--sample_id", type=str, default="sample", help="Sample ID for saving output")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    args = parser.parse_args()

    configs = load_configs(args.config)
    preprocess_config, model_config, train_config = configs
    results_dir = train_config["path"].get("result_path", "../Results/English")

    # Load Model
    model = get_model(args, configs, device, train=False)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        # handle module. prefix
        cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned_state)
    elif args.restore_step is None:
        # Check if checkpoints folder has any model
        ckpt_dir = train_config["path"]["ckpt_path"]
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth.tar") or f.endswith(".pt")]
            if ckpts:
                latest_ckpt = sorted(ckpts, key=lambda x: int(re.sub(r"\D", "", x) or 0))[-1]
                ckpt_path = os.path.join(ckpt_dir, latest_ckpt)
                print(f"Loading checkpoint automatically: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=device)
                state_dict = ckpt["model"] if "model" in ckpt else ckpt
                cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(cleaned_state)

    vocoder = get_vocoder(model_config, device)

    if args.test_set:
        print("Synthesizing official test sentences (E1, E2a, E2b, E3, E4)...")
        for sid, text in TEST_SENTENCES.items():
            synthesize_utterance(
                model, vocoder, text, sid, configs, results_dir,
                args.pitch_control, args.energy_control, args.duration_control
            )
    elif args.text:
        synthesize_utterance(
            model, vocoder, args.text, args.sample_id, configs, results_dir,
            args.pitch_control, args.energy_control, args.duration_control
        )
    else:
        print("Please provide --test_set to run all assignment sentences, or --text 'your sentence'.")


if __name__ == "__main__":
    main()
