import argparse
import os
import sys

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_configs(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    preprocess_config = config
    model_config = config.get("model", config)
    train_config = config
    return preprocess_config, model_config, train_config


def plot_loss_curves(train_losses, val_losses, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    
    # Plot training loss
    if train_losses:
        steps, losses = zip(*train_losses)
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, label="Training Total Loss", color="#1f77b4")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("FastSpeech 2 Training Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "training_loss.png"))
        plt.close()

    # Plot validation loss
    if val_losses:
        v_steps, v_losses = zip(*val_losses)
        plt.figure(figsize=(10, 6))
        plt.plot(v_steps, v_losses, label="Validation Total Loss", color="#ff7f0e", marker="o")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("FastSpeech 2 Validation Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "validation_loss.png"))
        plt.close()


def main(args, configs):
    print("Prepare training ...")
    preprocess_config, model_config, train_config = configs

    # Dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Model
    model, optimizer = get_model(args, configs, device, train=True)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Vocoder
    vocoder = get_vocoder(model_config, device)

    # Log directories
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    result_path = train_config["path"].get("result_path", "../Results/English")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    os.makedirs(train_config["path"]["ckpt_path"], exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    train_loss_history = []
    val_loss_history = []

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                output = model(*(batch[2:]))
                losses = Loss(batch, output)
                total_loss = losses[0]

                (total_loss / grad_acc_step).backward()
                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses_val = [l.item() for l in losses]
                    train_loss_history.append((step, losses_val[0]))
                    message = "Step {}/{}, Total: {:.4f}, Mel: {:.4f}, PostNet: {:.4f}, Pitch: {:.4f}, Energy: {:.4f}, Dur: {:.4f}".format(
                        step, total_step, *losses_val
                    )
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)
                    log(train_logger, step, losses=losses_val)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch, output, vocoder, model_config, preprocess_config
                    )
                    log(train_logger, fig=fig, tag="Training/step_{}_{}".format(step, tag))
                    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
                    log(train_logger, audio=wav_reconstruction, sampling_rate=sampling_rate, tag="Training/step_{}_{}_rec".format(step, tag))
                    log(train_logger, audio=wav_prediction, sampling_rate=sampling_rate, tag="Training/step_{}_{}_synth".format(step, tag))

                if step % val_step == 0:
                    model.eval()
                    eval_msg, val_total_loss = evaluate(model, step, configs, val_logger, vocoder)
                    val_loss_history.append((step, val_total_loss))
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(eval_msg + "\n")
                    outer_bar.write(eval_msg)
                    plot_loss_curves(train_loss_history, val_loss_history, result_path)
                    model.train()

                if step % save_step == 0:
                    save_model = model.module if hasattr(model, "module") else model
                    torch.save(
                        {
                            "model": save_model.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(train_config["path"]["ckpt_path"], "{}.pth.tar".format(step)),
                    )
                    plot_loss_curves(train_loss_history, val_loss_history, result_path)

                if step >= total_step:
                    plot_loss_curves(train_loss_history, val_loss_history, result_path)
                    print("Training completed!")
                    return

                step += 1
                outer_bar.update(1)
            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to unified config.yaml")
    parser.add_argument("-p", "--preprocess_config", type=str, default=None)
    parser.add_argument("-m", "--model_config", type=str, default=None)
    parser.add_argument("-t", "--train_config", type=str, default=None)
    args = parser.parse_args()

    if args.preprocess_config and args.model_config and args.train_config:
        preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
        model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
        train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
        configs = (preprocess_config, model_config, train_config)
    else:
        configs = load_configs(args.config)

    main(args, configs)
