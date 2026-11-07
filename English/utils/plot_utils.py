import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_comprehensive_losses(train_history, val_history, result_dir):
    """
    Plots extensive visual loss curves:
    1. training_loss.png
    2. validation_loss.png
    3. loss_breakdown.png (6-panel subplot: Total, Mel, PostNet Mel, Duration, Pitch, Energy)
    4. variance_losses.png (Duration, Pitch, Energy comparison)
    """
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "eval_plots"), exist_ok=True)

    # 1. Standard Training Loss
    if train_history:
        steps = [h[0] for h in train_history]
        total_train = [h[1][0] for h in train_history]
        plt.figure(figsize=(9, 5))
        plt.plot(steps, total_train, label="Train Total Loss", color="#1f77b4", linewidth=1.5)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("FastSpeech 2 Training Loss", fontsize=14, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "training_loss.png"), dpi=200)
        plt.close()

    # 2. Standard Validation Loss
    if val_history:
        v_steps = [h[0] for h in val_history]
        total_val = [h[1][0] for h in val_history]
        plt.figure(figsize=(9, 5))
        plt.plot(v_steps, total_val, label="Validation Total Loss", color="#ff7f0e", marker="o", linewidth=1.5)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("FastSpeech 2 Validation Loss", fontsize=14, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "validation_loss.png"), dpi=200)
        plt.close()

    # 3. 6-Panel Loss Breakdown (Train vs Validation across all components)
    if train_history and val_history:
        steps = [h[0] for h in train_history]
        v_steps = [h[0] for h in val_history]
        
        loss_names = ["Total Loss", "Mel Loss", "PostNet Mel Loss", "Pitch Loss", "Energy Loss", "Duration Loss"]
        colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx in range(6):
            ax = axes[idx]
            train_vals = [h[1][idx] for h in train_history]
            val_vals = [h[1][idx] for h in val_history]
            
            ax.plot(steps, train_vals, label="Train", color=colors[idx], alpha=0.85, linewidth=1.2)
            ax.plot(v_steps, val_vals, label="Validation", color="#d62728", linestyle="--", marker="o", markersize=3)
            ax.set_title(loss_names[idx], fontsize=13, fontweight="bold")
            ax.set_xlabel("Steps", fontsize=10)
            ax.set_ylabel("Loss", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=10)
            
        plt.suptitle("FastSpeech 2 Comprehensive Multi-Component Loss Breakdown", fontsize=16, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(result_dir, "loss_breakdown.png"), dpi=200)
        plt.close()

    # 4. Variance Predictor Components Plot
    if train_history:
        steps = [h[0] for h in train_history]
        pitch_loss = [h[1][3] for h in train_history]
        energy_loss = [h[1][4] for h in train_history]
        dur_loss = [h[1][5] for h in train_history]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, pitch_loss, label="Pitch Loss", color="#9467bd", linewidth=1.3)
        plt.plot(steps, energy_loss, label="Energy Loss", color="#8c564b", linewidth=1.3)
        plt.plot(steps, dur_loss, label="Duration Loss", color="#e377c2", linewidth=1.3)
        plt.xlabel("Step", fontsize=11)
        plt.ylabel("Loss Value", fontsize=11)
        plt.title("Variance Adaptor Component Convergence (Pitch, Energy, Duration)", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "variance_losses.png"), dpi=200)
        plt.close()

def plot_validation_contours(targets, predictions, step, result_dir):
    """
    Plots predicted vs ground truth pitch and energy contours during validation.
    """
    eval_dir = os.path.join(result_dir, "eval_plots")
    os.makedirs(eval_dir, exist_ok=True)
    
    try:
        src_len = predictions[8][0].item()
        mel_len = predictions[9][0].item()
        dur_gt = targets[11][0, :src_len].detach().cpu().numpy()
        dur_pred = predictions[5][0, :src_len].detach().cpu().numpy()
        
        # Duration comparison bar plot
        plt.figure(figsize=(12, 4))
        x = np.arange(min(src_len, 40))
        plt.bar(x - 0.2, dur_gt[:40], width=0.4, label="Ground Truth Duration", color="#1f77b4", alpha=0.7)
        plt.bar(x + 0.2, dur_pred[:40], width=0.4, label="Predicted Duration", color="#ff7f0e", alpha=0.7)
        plt.xlabel("Phoneme Index", fontsize=11)
        plt.ylabel("Frames", fontsize=11)
        plt.title(f"Step {step}: Predicted vs Ground Truth Duration per Phoneme", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, f"duration_eval_step_{step}.png"), dpi=180)
        plt.close()
    except Exception as e:
        pass
