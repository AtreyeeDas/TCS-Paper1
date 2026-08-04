#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
temporal_boundary_analysis.py

A true temporal trajectory experiment for XTTS-v2 GPT cross-attention.
Investigates the step-by-step evolution of INDIVIDUAL significant decoder heads 
as they approach, cross, and leave a language-switch boundary in Decoder Time (±40 steps).
"""

import os
import sys
import glob
import pickle
import logging
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings

# Suppress minor warnings for clean logs
warnings.filterwarnings("ignore", category=RuntimeWarning)

# XTTS specific imports
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer


# ==========================================================
# CONFIGURATION & LOGGING
# ==========================================================
def setup_directories(base_dir):
    """Creates the strictly required directory structure."""
    dirs = {
        'trajectories': os.path.join(base_dir, 'trajectories'),
        'heatmaps': os.path.join(base_dir, 'heatmaps'),
        'validation': os.path.join(base_dir, 'validation'),
        'supplementary': os.path.join(base_dir, 'supplementary'),
        'logs': os.path.join(base_dir, 'logs'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(dirs['logs'], 'temporal_analysis.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return dirs


# ==========================================================
# TOKENIZER & BOUNDARY DETECTION
# ==========================================================
def load_xtts_tokenizer(config_path):
    logging.info(f"Loading XTTS tokenizer from {config_path}")
    try:
        config = XttsConfig()
        config.load_json(config_path)
        vocab_path = os.path.join(os.path.dirname(config_path), "vocab.json")
        return VoiceBpeTokenizer(vocab_file=vocab_path)
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

def classify_token(text):
    if not text: return "SPACE"
    special = ["[START]", "[STOP]", "[en]", "[hi]", "[zh]"]
    if text in special or (text.startswith("[") and text.endswith("]")): return "SPECIAL"
    if text.isspace() or text == "_" or text == " ": return "SPACE"
    if any('\u0900' <= c <= '\u097F' for c in text): return "HINDI"
    if any('a' <= c.lower() <= 'z' for c in text): return "ENGLISH"
    return "OTHER"

def detect_boundaries(token_classes):
    boundaries = []
    last_lang = None
    for i, cls in enumerate(token_classes):
        if cls in ["ENGLISH", "HINDI"]:
            if last_lang is not None and cls != last_lang:
                boundaries.append(i) # Tag transition token
            last_lang = cls
    return boundaries


# ==========================================================
# EXACT METRIC EXTRACTION
# ==========================================================
def compute_metrics(A):
    """Computes the 15 exact metrics over Decoder Time (T)."""
    T, L, H, K = A.shape
    A = np.clip(A, 1e-12, 1.0)
    
    metrics = {}
    metrics['Entropy'] = -np.sum(A * np.log2(A), axis=3)
    metrics['Peak Probability'] = np.max(A, axis=3)
    
    k_indices = np.arange(K).reshape(1, 1, 1, K)
    com = np.sum(A * k_indices, axis=3)
    metrics['Centre of Mass'] = com
    
    diff = k_indices - com[..., np.newaxis]
    metrics['Variance'] = np.sum(A * (diff ** 2), axis=3)
    metrics['Width'] = np.sum(A > 0.05, axis=3)
    
    t_indices = np.arange(T).reshape(T, 1, 1)
    expected_com = (t_indices / max(T-1, 1)) * (K - 1)
    metrics['Diagonal Deviation'] = np.abs(com - expected_com)
    
    jump, vel, accel = np.zeros((T,L,H)), np.zeros((T,L,H)), np.zeros((T,L,H))
    jump[1:] = np.abs(com[1:] - com[:-1])
    vel[1:] = com[1:] - com[:-1]
    accel[1:] = vel[1:] - vel[:-1]
    metrics['Jump Distance'] = jump
    metrics['Velocity'] = vel
    metrics['Acceleration'] = accel
    
    kl = np.zeros((T, L, H))
    A_prev = np.clip(np.roll(A, shift=1, axis=0), 1e-12, 1.0)
    kl[1:] = np.sum(A[1:] * np.log2(A[1:] / A_prev[1:]), axis=3)
    metrics['KL Divergence'] = kl
    
    cdf, cdf_prev = np.cumsum(A, axis=3), np.roll(np.cumsum(A, axis=3), shift=1, axis=0)
    emd = np.zeros((T, L, H))
    emd[1:] = np.sum(np.abs(cdf[1:] - cdf_prev[1:]), axis=3)
    metrics['EMD'] = emd
    
    metrics['Sharpness'] = np.sum(A ** 2, axis=3)
    metrics['Sparsity'] = np.sum(A < 0.01, axis=3) / K
    metrics['Concentration'] = np.sum(np.sort(A, axis=3)[..., -3:], axis=3)
    
    ent_deriv = np.zeros((T, L, H))
    ent_deriv[1:] = metrics['Entropy'][1:] - metrics['Entropy'][:-1]
    metrics['Entropy Derivative'] = ent_deriv
    
    return metrics


# ==========================================================
# VALIDATION AND ALIGNMENT
# ==========================================================
def extract_tensors(utterance_dir, tokenizer):
    """Loads A tensor and boundary locations for one utterance."""
    attention_path = os.path.join(utterance_dir, "generation_attentions.pkl")
    tokens_path = os.path.join(utterance_dir, "text_tokens.pt")
    if not os.path.exists(attention_path) or not os.path.exists(tokens_path): 
        return None

    text_tokens = torch.load(tokens_path, map_location="cpu").squeeze()
    text_len = text_tokens.shape[0]
    decoded_tokens = [tokenizer.decode([tid]) for tid in text_tokens.tolist()]
    token_classes = [classify_token(t) for t in decoded_tokens]
    boundaries = detect_boundaries(token_classes)
    
    if not boundaries: 
        return None

    with open(attention_path, "rb") as f: attentions = pickle.load(f)
    num_gen_steps = len(attentions)
    if num_gen_steps == 0: return None
        
    audio_cond_len = attentions[0][0].shape[-1] - text_len
    L, H = len(attentions[0]), attentions[0][0].shape[1]
    
    A = np.zeros((num_gen_steps, L, H, text_len))
    for t in range(num_gen_steps):
        for l in range(L):
            last_q = attentions[t][l][0, :, -1, :]
            t_keys = last_q[:, audio_cond_len : audio_cond_len + text_len].float().cpu().numpy()
            row_sums = t_keys.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            A[t, l, :, :] = t_keys / row_sums

    raw_metrics = compute_metrics(A)
    return {
        'utt_id': os.path.basename(os.path.normpath(utterance_dir)),
        'A': A,
        'raw_metrics': raw_metrics,
        'boundaries': boundaries,
        'decoded_tokens': decoded_tokens,
        'total_steps': num_gen_steps
    }


def perform_validation_and_extraction(results_list, target_heads, dirs):
    """
    Finds Decoder Time Zero, validates alignment mathematically, 
    and extracts ±40 step metric windows.
    """
    logging.info("Validating Decoder-Time Alignment and computing Time Zero...")
    
    validation_records = []
    window_size = 40
    metric_names = list(results_list[0]['raw_metrics'].keys())
    
    # Trajectories storage: dict[head][metric] = list of 81-step arrays
    trajectories = {(l, h): {m: [] for m in metric_names} for (l, h) in target_heads}
    
    unreliable_count = 0
    total_checks = 0

    for res in results_list:
        A = res['A'] # (T, L, H, K)
        utt_id = res['utt_id']
        total_steps = res['total_steps']
        
        for b_idx in res['boundaries']:
            b_token_str = res['decoded_tokens'][b_idx]
            
            for (l, h) in target_heads:
                total_checks += 1
                
                # Decoder Time Zero = generation step where this head paid max attention to boundary token
                head_attn_to_b = A[:, l, h, b_idx]
                time_zero = int(np.argmax(head_attn_to_b))
                max_attn_val = head_attn_to_b[time_zero]
                
                if max_attn_val < 0.01:
                    unreliable_count += 1
                
                step_start = time_zero - window_size
                step_end = time_zero + window_size + 1
                
                validation_records.append({
                    'Utterance ID': utt_id,
                    'Layer': l,
                    'Head': h,
                    'Boundary Token': b_token_str,
                    'Token Index': b_idx,
                    'Decoder Time Zero': time_zero,
                    'Step Range Extracted': f"[{step_start}, {step_end-1}]",
                    'Total Decoder Steps': total_steps,
                    'Max Attention': max_attn_val
                })
                
                # Extract the ±40 windows for all metrics
                for m_name in metric_names:
                    full_metric = res['raw_metrics'][m_name][:, l, h]
                    
                    window = np.full(81, np.nan)
                    # Calculate valid overlapping bounds
                    v_start = max(0, step_start)
                    v_end = min(total_steps, step_end)
                    
                    w_start = v_start - step_start
                    w_end = w_start + (v_end - v_start)
                    
                    if v_start < v_end:
                        window[w_start:w_end] = full_metric[v_start:v_end]
                        
                    trajectories[(l, h)][m_name].append(window)

    # -------------------------------------------------------------
    # MATHEMATICAL VALIDATION ABORT CHECK
    # -------------------------------------------------------------
    if (unreliable_count / max(total_checks, 1)) > 0.30:
        logging.error("\n" + "="*60)
        logging.error("ALIGNMENT VALIDATION FAILED")
        logging.error("="*60)
        logging.error("Decoder Time Zero cannot be reliably estimated.")
        logging.error("Reason: Over 30% of the significant heads failed to attend (A < 0.01) ")
        logging.error("to the exact boundary token during the autoregressive generation loop.")
        logging.error("This implies the selected heads operate on distributed contextual representations")
        logging.error("rather than discrete token-aligned mechanisms. Producing specific temporal")
        logging.error("heatmaps around 'Time Zero' would be scientifically invalid.")
        logging.error("Aborting to prevent fabrication of results.")
        sys.exit(1)

    # Save validation records
    df_val = pd.DataFrame(validation_records)
    df_val.to_csv(os.path.join(dirs['validation'], 'alignment_validation.csv'), index=False)
    
    # Print 5 random validations
    print("\n" + "="*60)
    print("DECODER-TIME ALIGNMENT VALIDATION (5 Random Samples)")
    print("="*60)
    samples = df_val.sample(n=min(5, len(df_val)))
    for _, row in samples.iterrows():
        print(f"Utterance: {row['Utterance ID']} | Head: L{row['Layer']}_H{row['Head']}")
        print(f"  -> Boundary Token: '{row['Boundary Token']}' (Idx: {row['Token Index']})")
        print(f"  -> Decoder Time Zero: Step {row['Decoder Time Zero']} (Max A: {row['Max Attention']:.3f})")
        print(f"  -> Extraction Range: {row['Step Range Extracted']} out of {row['Total Decoder Steps']} total steps\n")
        
    with open(os.path.join(dirs['validation'], 'alignment_examples.txt'), 'w') as f:
        f.write(samples.to_string())

    return trajectories


# ==========================================================
# PLOTTING FUNCTIONS
# ==========================================================
def plot_head_metric_heatmaps(trajectories, target_heads, metric_names, dirs):
    """
    Generates 1 Heatmap per individual Head.
    Rows: Metrics. Columns: Relative Time (-40 to +40).
    """
    logging.info("Generating Head-Specific Metric Evolution Heatmaps...")
    x_axis = np.arange(-40, 41)
    
    for (l, h) in target_heads:
        # Build matrix (15 rows, 81 cols)
        matrix = np.zeros((len(metric_names), 81))
        
        for i, m in enumerate(metric_names):
            # Average over all boundaries for this head
            mean_traj = np.nanmean(trajectories[(l, h)][m], axis=0)
            
            # Normalize row to [0, 1] for relative comparison
            t_min, t_max = np.nanmin(mean_traj), np.nanmax(mean_traj)
            if t_max - t_min > 0:
                norm_traj = (mean_traj - t_min) / (t_max - t_min)
            else:
                norm_traj = np.zeros_like(mean_traj)
                
            matrix[i, :] = norm_traj
            
        plt.figure(figsize=(14, 8))
        sns.heatmap(matrix, cmap="magma", yticklabels=metric_names, 
                    xticklabels=[x if x % 10 == 0 else "" for x in x_axis])
        plt.axvline(x=40.5, color='white', linestyle='--', linewidth=2, label='Time Zero')
        
        plt.title(f"Complete Metric Evolution over Decoder Time (Layer {l}, Head {h})")
        plt.xlabel("Relative Decoder Generation Step")
        plt.ylabel("Normalized Metric Trajectory [0, 1]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dirs['heatmaps'], f"head_L{l:02d}_H{h:02d}_metric_heatmap.png"), dpi=300)
        plt.close()


def plot_individual_trajectories(trajectories, target_heads, metric_names, dirs):
    """Generates pure temporal line graphs with 95% CI."""
    logging.info("Generating Individual Trajectory Plots...")
    x_axis = np.arange(-40, 41)
    
    for (l, h) in target_heads:
        for m in metric_names:
            data = np.array(trajectories[(l, h)][m], dtype=float)
            
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
            counts = np.sum(~np.isnan(data), axis=0)
            se = np.divide(std, np.sqrt(counts), out=np.zeros_like(std), where=counts>1)
            
            plt.figure(figsize=(8, 5))
            plt.plot(x_axis, mean, color='blue', linewidth=2, label='Mean Trajectory')
            plt.fill_between(x_axis, mean - 1.96*se, mean + 1.96*se, color='blue', alpha=0.2, label='95% CI')
            plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label="Boundary (Time Zero)")
            
            plt.title(f"Temporal Trajectory of {m}\nLayer {l}, Head {h}")
            plt.xlabel("Relative Decoder Generation Step")
            plt.ylabel(f"Absolute {m}")
            plt.xticks(np.arange(-40, 41, 10))
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(dirs['trajectories'], f"head_L{l:02d}_H{h:02d}_{m.replace(' ', '_')}.png"), dpi=300)
            plt.close()


def plot_supplementary_average(trajectories, target_heads, metric_names, dirs):
    """Generates an explicit supplementary averaged heatmap."""
    logging.info("Generating Supplementary Head-Averaged Analysis...")
    x_axis = np.arange(-40, 41)
    matrix = np.zeros((len(metric_names), 81))
    
    for i, m in enumerate(metric_names):
        # Pool all trajectories across all Top 10 heads
        pooled = []
        for (l, h) in target_heads:
            pooled.extend(trajectories[(l, h)][m])
            
        mean_traj = np.nanmean(pooled, axis=0)
        t_min, t_max = np.nanmin(mean_traj), np.nanmax(mean_traj)
        matrix[i, :] = (mean_traj - t_min) / (t_max - t_min + 1e-9)

    plt.figure(figsize=(14, 8))
    sns.heatmap(matrix, cmap="viridis", yticklabels=metric_names, 
                xticklabels=[x if x % 10 == 0 else "" for x in x_axis])
    plt.axvline(x=40.5, color='white', linestyle='--', linewidth=2, label='Time Zero')
    
    plt.title("Supplementary Head-Averaged Analysis (Top 10 Heads)\nNOTE: For Observational Use Only - Not for Primary Scientific Conclusions")
    plt.xlabel("Relative Decoder Generation Step")
    plt.ylabel("Normalized Metric Trajectory [0, 1]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['supplementary'], "head_averaged_heatmap.png"), dpi=300)
    plt.close()


# ==========================================================
# MAIN
# ==========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./xtts_attention_results")
    parser.add_argument("--config_path", type=str, default="/home/spark2/Models/XTTS-v2/config.json")
    parser.add_argument("--sig_heads", type=str, default="./corrected_analysis_results/significant_heads.csv")
    parser.add_argument("--output_dir", type=str, default="./temporal_boundary_results")
    args = parser.parse_args()

    dirs = setup_directories(args.output_dir)
    
    if not os.path.exists(args.sig_heads):
        logging.error(f"CRITICAL ERROR: {args.sig_heads} not found. Run Experiment 2 first.")
        sys.exit(1)

    # Select Top 10 Significant Heads
    df_sig = pd.read_csv(args.sig_heads)
    if df_sig.empty:
        logging.error("significant_heads.csv is empty.")
        sys.exit(1)
        
    df_sig['abs_d'] = df_sig['cohens_d'].abs()
    df_sig = df_sig.sort_values(by=['p_adj', 'abs_d'], ascending=[True, False])
    target_heads = [(int(row['layer']), int(row['head'])) for _, row in df_sig.head(10).iterrows()]
    
    logging.info(f"Primary analysis restricted to Top {len(target_heads)} significant heads.")

    tokenizer = load_xtts_tokenizer(args.config_path)
    utterance_dirs = [d for d in glob.glob(os.path.join(args.data_dir, "*_*")) if os.path.isdir(d)]

    logging.info("Extracting Raw Attention Tensors...")
    results_list = []
    for u_dir in utterance_dirs:
        res = extract_tensors(u_dir, tokenizer)
        if res: results_list.append(res)
            
    if not results_list:
        logging.error("No valid boundaries detected.")
        sys.exit(1)
        
    # Validation & Trajectory Extraction
    trajectories = perform_validation_and_extraction(results_list, target_heads, dirs)
    metric_names = list(results_list[0]['raw_metrics'].keys())

    # Plotting
    plot_head_metric_heatmaps(trajectories, target_heads, metric_names, dirs)
    plot_individual_trajectories(trajectories, target_heads, metric_names, dirs)
    plot_supplementary_average(trajectories, target_heads, metric_names, dirs)

    logging.info(f"Experiment completed. All data strictly segregated by head in {args.output_dir}.")

if __name__ == "__main__":
    main()
