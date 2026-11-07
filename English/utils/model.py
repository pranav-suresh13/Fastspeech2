import os
import sys
import json

import numpy as np
import torch

# PyTorch 2.6+ unpickler safety override
try:
    torch.serialization.add_safe_globals([
        np.core.multiarray.scalar,
        np.ndarray,
        np.dtype,
        getattr(np.core.multiarray, '_reconstruct', None)
    ])
except Exception:
    pass

import hifigan
from model import FastSpeech2, ScheduledOptim


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        ckpt_path = "hifigan/generator_universal.pth.tar" if speaker == "universal" else "hifigan/generator_LJSpeech.pth.tar"
        if not os.path.exists(ckpt_path):
            print(f"[Notice] Vocoder checkpoint '{ckpt_path}' not found. Audio logging will be skipped.")
            return None
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            try:
                ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar", map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar", map_location=device)
        elif speaker == "universal":
            try:
                ckpt = torch.load("hifigan/generator_universal.pth.tar", map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load("hifigan/generator_universal.pth.tar", map_location=device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i] * hop_length]

    return wavs
