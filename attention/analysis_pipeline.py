#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis_pipeline.py

Complete pipeline for analyzing autoregressive attention entropy around 
code-switch boundaries (English <-> Hindi) in Zero-Shot CS-TTS.
Designed for Coqui XTTS-v2 HuggingFace GPT2 backbone.
"""

import os
import glob
import json
import pickle
import argparse
import logging
import warnings
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# XTTS specific imports
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

# Suppress minor warnings for clean ICASSP logs
warnings.filterwarnings("ignore", category=RuntimeWarning)


def setup_logging(output_dir):
    """Initializes logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'analysis_pipeline.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Results will be saved to {output_dir}")


def load_xtts_tokenizer(config_path):
    """Reloads the XTTS tokenizer using the config and vocabulary."""
    logging.info(f"Loading XTTS tokenizer from {config_path}")
    try:
        config = XttsConfig()
        config.load_json(config_path)
        vocab_path = os.path.join(os.path.dirname(config_path), "vocab.json")
        tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
        return tokenizer
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        raise


def classify_token(text):
    """
    Classifies a decoded token string into ENGLISH, HINDI, SPACE, SPECIAL, or OTHER
    using precise Unicode block detection.
    """
    if not text:
        return "SPACE"
    
    # Check for XTTS/HuggingFace special tokens
    special_tokens = ["[START]", "[STOP]", "[en]", "[hi]", "[zh]"]
    if text in special_tokens or (text.startswith("[") and text.endswith("]")):
        return "SPECIAL"
        
    if text.isspace() or text == "_" or text == " ":
        return "SPACE"
        
    # Devanagari Unicode Block: \u0900 - \u097F
    if any('\u0900' <= c <= '\u097F' for c in text):
        return "HINDI"
        
    # Basic Latin Block
    if any('a' <= c.lower() <= 'z' for c in text):
        return "ENGLISH"
        
    return "OTHER"


def detect_boundaries(token_classes):
    """
    Detects English <-> Hindi transitions.
    Returns a list of tuples: (idx_before_boundary, idx_after_boundary)
    """
    boundaries = []
    last_lang = None
    last_lang_idx = -1
    
    for i, cls in enumerate(token_classes):
        if cls in ["ENGLISH", "HINDI"]:
            if last_lang is not None and cls != last_lang:
                boundaries.append((last_lang_idx, i))
            last_lang = cls
            last_lang_idx = i
            
    return boundaries


def compute_shannon_entropy(probs):
    """
    Computes Shannon entropy: H(P) = -sum(P * log2(P))
    probs: numpy array of shape (N,)
    """
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def process_utterance(utterance_dir, tokenizer, window_size=5):
    """
    Processes a single utterance directory.
    Reconstructs the GenerationStep x KeyPosition matrix, aligns audio tokens to text,
    and extracts boundary vs. neighbour entropies.
    """
    attention_path = os.path.join(utterance_dir, "generation_attentions.pkl")
    tokens_path = os.path.join(utterance_dir, "text_tokens.pt")
    
    if not os.path.exists(attention_path) or not os.path.exists(tokens_path):
        logging.warning(f"Missing required files in {utterance_dir}. Skipping.")
        return None

    # Load text tokens
    text_tokens = torch.load(tokens_path, map_location="cpu")
    if text_tokens.dim() > 1:
        text_tokens = text_tokens.squeeze(0)
    text_len = text_tokens.shape[0]
    
    # Decode and classify
    decoded_tokens = []
    token_classes = []
    for token_id in text_tokens.tolist():
        text_str = tokenizer.decode([token_id])
        cls = classify_token(text_str)
        decoded_tokens.append(text_str)
        token_classes.append(cls)
        
    boundaries = detect_boundaries(token_classes)
    if not boundaries:
        return None # Skip utterances with no code-switching
        
    # Load generation attentions
    with open(attention_path, "rb") as f:
        attentions = pickle.load(f)
        
    num_gen_steps = len(attentions)
    if num_gen_steps == 0:
        return None
        
    # Determine the audio condition length to isolate text keys
    # attention shape at t=0: (batch, heads, q_len, k_len)
    first_step_attn = attentions[0][0]
    total_prompt_len = first_step_attn.shape[-1]
    
    if total_prompt_len < text_len:
        logging.warning(f"Total prompt length ({total_prompt_len}) < text length ({text_len}) in {utterance_dir}. Skipping.")
        return None
        
    audio_cond_len = total_prompt_len - text_len
    
    # Reconstruct GenerationStep x TextKeyPosition matrix (Head & Layer Averaged)
    gen_text_attn = np.zeros((num_gen_steps, text_len))
    
    for t in range(num_gen_steps):
        step_layers = attentions[t]
        layer_tensors = []
        for layer_attn in step_layers:
            # layer_attn shape: (batch, heads, q, k)
            # We take the last query vector (the currently generated token)
            last_q_attn = layer_attn[0, :, -1, :] # shape (heads, k)
            
            # Slice out only the text token keys
            text_keys_attn = last_q_attn[:, audio_cond_len : audio_cond_len + text_len]
            layer_tensors.append(text_keys_attn)
            
        # Stack and average across layers and heads
        step_tensor = torch.stack(layer_tensors) # (num_layers, num_heads, text_len)
        step_avg = step_tensor.mean(dim=(0, 1)).float().cpu().numpy()
        
        # Normalize to form a valid probability distribution over text tokens
        step_sum = step_avg.sum()
        if step_sum > 0:
            step_avg = step_avg / step_sum
        else:
            step_avg = np.ones_like(step_avg) / len(step_avg)
            
        gen_text_attn[t, :] = step_avg

    # Compute entropy for each generation step
    step_entropies = np.array([compute_shannon_entropy(gen_text_attn[t, :]) for t in range(num_gen_steps)])
    
    # Align text tokens to generation steps (argmax mapping)
    # text_token_entropies[i] will store all step entropies that maximally attended to text token i
    alignment_mapping = defaultdict(list)
    for t in range(num_gen_steps):
        max_attended_text_idx = np.argmax(gen_text_attn[t, :])
        alignment_mapping[max_attended_text_idx].append(step_entropies[t])
        
    text_token_entropies = np.full(text_len, np.nan)
    for idx in range(text_len):
        if idx in alignment_mapping and len(alignment_mapping[idx]) > 0:
            text_token_entropies[idx] = np.mean(alignment_mapping[idx])
            
    # Interpolate NaNs for text tokens that received no maximal attention
    df_interp = pd.Series(text_token_entropies).interpolate(limit_direction='both')
    text_token_entropies = df_interp.to_numpy()

    # Extract Boundary, Neighbour, and Global entropies
    global_entropy = np.nanmean(text_token_entropies)
    
    boundary_entropies = []
    neighbour_entropies = []
    
    for (idx1, idx2) in boundaries:
        b_val1 = text_token_entropies[idx1]
        b_val2 = text_token_entropies[idx2]
        
        if not np.isnan(b_val1) and not np.isnan(b_val2):
            boundary_entropies.append((b_val1 + b_val2) / 2.0)
            
        # Local window
        start_idx = max(0, min(idx1, idx2) - window_size)
        end_idx = min(text_len, max(idx1, idx2) + window_size + 1)
        
        n_vals = []
        for i in range(start_idx, end_idx):
            if i != idx1 and i != idx2 and not np.isnan(text_token_entropies[i]):
                n_vals.append(text_token_entropies[i])
                
        if n_vals:
            neighbour_entropies.append(np.mean(n_vals))

    if not boundary_entropies or not neighbour_entropies:
        return None

    # We return the mean for this utterance to do utterance-level statistics
    result = {
        "utt_id": os.path.basename(os.path.normpath(utterance_dir)),
        "boundary_entropy": np.mean(boundary_entropies),
        "neighbour_entropy": np.mean(neighbour_entropies),
        "global_entropy": global_entropy,
        "num_boundaries": len(boundaries),
        "matrix": gen_text_attn,
        "text_token_entropies": text_token_entropies,
        "boundaries": boundaries,
        "decoded_tokens": decoded_tokens
    }
    
    return result


def compute_statistics(df):
    """
    Performs rigorous statistical testing (Paired t-test, Wilcoxon, Cohen's d).
    """
    b_ent = df['boundary_entropy'].to_numpy()
    n_ent = df['neighbour_entropy'].to_numpy()
    
    # Paired t-test
    t_stat, p_val_t = stats.ttest_rel(b_ent, n_ent)
    
    # Wilcoxon signed-rank test
    w_stat, p_val_w = stats.wilcoxon(b_ent, n_ent)
    
    # Cohen's d
    diff = b_ent - n_ent
    effect_size = np.mean(diff) / np.std(diff, ddof=1)
    
    stats_results = {
        "avg_boundary": np.mean(b_ent),
        "avg_neighbour": np.mean(n_ent),
        "avg_global": df['global_entropy'].mean(),
        "t_stat": t_stat,
        "p_val_t": p_val_t,
        "w_stat": w_stat,
        "p_val_w": p_val_w,
        "cohens_d": effect_size,
        "total_boundaries": df['num_boundaries'].sum(),
        "total_utterances": len(df)
    }
    
    return stats_results


def generate_figures(results_list, df, output_dir):
    """
    Generates and saves the 5 required ICASSP figures.
    """
    sns.set_theme(style="whitegrid", context="paper")
    
    # Select the first valid utterance for sequence-specific plots (A and B)
    sample = results_list[0]
    
    # A. Heatmap with language boundary overlaid
    plt.figure(figsize=(10, 6))
    sns.heatmap(sample["matrix"].T, cmap="viridis", cbar_kws={'label': 'Attention Weight'})
    plt.title(f"Decoder Attention over Text Tokens\n{sample['utt_id']}")
    plt.xlabel("Autoregressive Generation Step")
    plt.ylabel("Text Token Index")
    # Overlay boundaries
    for (idx1, idx2) in sample["boundaries"]:
        b_idx = (idx1 + idx2) / 2.0
        plt.axhline(y=b_idx, color='red', linestyle='--', linewidth=2, label='CS Boundary' if b_idx == (sample["boundaries"][0][0]+sample["boundaries"][0][1])/2 else "")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "A_attention_heatmap.png"), dpi=300)
    plt.close()
    
    # B. Entropy curve with boundary marker
    plt.figure(figsize=(10, 4))
    x_axis = np.arange(len(sample["text_token_entropies"]))
    plt.plot(x_axis, sample["text_token_entropies"], marker='o', linestyle='-', color='b')
    plt.title(f"Attention Entropy aligned to Text Tokens\n{sample['utt_id']}")
    plt.xlabel("Text Token Index")
    plt.ylabel("Shannon Entropy (bits)")
    for (idx1, idx2) in sample["boundaries"]:
        b_idx = (idx1 + idx2) / 2.0
        plt.axvline(x=b_idx, color='red', linestyle='--', linewidth=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "B_entropy_curve.png"), dpi=300)
    plt.close()
    
    # C. Average entropy over all utterances (Bar chart)
    plt.figure(figsize=(6, 5))
    means = [df['boundary_entropy'].mean(), df['neighbour_entropy'].mean(), df['global_entropy'].mean()]
    std_errs = [stats.sem(df['boundary_entropy']), stats.sem(df['neighbour_entropy']), stats.sem(df['global_entropy'])]
    sns.barplot(x=['Boundary', 'Neighbour (±5)', 'Global'], y=means, capsize=0.1, errorbar=None)
    plt.errorbar(x=[0, 1, 2], y=means, yerr=std_errs, fmt='none', c='black', capsize=5)
    plt.title("Average Attention Entropy")
    plt.ylabel("Shannon Entropy (bits)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "C_average_entropy.png"), dpi=300)
    plt.close()
    
    # D. Boundary vs Non-boundary boxplot
    plt.figure(figsize=(6, 5))
    melted_df = df.melt(value_vars=['boundary_entropy', 'neighbour_entropy'], 
                        var_name='Region', value_name='Entropy')
    melted_df['Region'] = melted_df['Region'].map({'boundary_entropy': 'Boundary', 'neighbour_entropy': 'Neighbour (±5)'})
    sns.boxplot(x='Region', y='Entropy', data=melted_df, palette="Set2")
    plt.title("Entropy Distribution: Boundary vs Neighbour")
    plt.ylabel("Shannon Entropy (bits)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "D_boundary_vs_neighbour_boxplot.png"), dpi=300)
    plt.close()
    
    # E. Histogram of entropy difference
    plt.figure(figsize=(7, 5))
    diff = df['boundary_entropy'] - df['neighbour_entropy']
    sns.histplot(diff, bins=20, kde=True, color='purple')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title("Histogram of Entropy Differences (Boundary - Neighbour)")
    plt.xlabel("Entropy Difference (bits)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "E_entropy_difference_hist.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Extracts and analyzes XTTS GPT cross-attention for CS-TTS Boundaries.")
    parser.add_argument("--data_dir", type=str, default="./xtts_attention_results", help="Directory containing the utterance folders.")
    parser.add_argument("--config_path", type=str, default="/home/spark2/Models/XTTS-v2/config.json", help="Path to XTTS config.json.")
    parser.add_argument("--output_dir", type=str, default="./analysis_results", help="Directory to save plots and CSVs.")
    parser.add_argument("--window_size", type=int, default=5, help="Tokens to include in the neighbour window.")
    args = parser.parse_args()

    setup_logging(args.output_dir)
    
    # 1. Load Tokenizer
    tokenizer = load_xtts_tokenizer(args.config_path)
    
    # 2. Iterate over utterances
    utterance_dirs = glob.glob(os.path.join(args.data_dir, "*_*"))
    utterance_dirs = [d for d in utterance_dirs if os.path.isdir(d)]
    
    if not utterance_dirs:
        logging.error(f"No utterance directories found in {args.data_dir}")
        return
        
    logging.info(f"Found {len(utterance_dirs)} utterances. Commencing analysis...")
    
    results_list = []
    
    for u_dir in tqdm(utterance_dirs, desc="Processing Attentions"):
        res = process_utterance(u_dir, tokenizer, window_size=args.window_size)
        if res is not None:
            results_list.append(res)
            # Save the raw numpy matrix for reproducibility
            np.save(os.path.join(args.output_dir, f"{res['utt_id']}_attn_matrix.npy"), res["matrix"])
            
    if not results_list:
        logging.error("No valid code-switched boundaries detected across the dataset.")
        return
        
    # 3. Create DataFrame and Compute Stats
    df = pd.DataFrame([{
        "utt_id": r["utt_id"],
        "boundary_entropy": r["boundary_entropy"],
        "neighbour_entropy": r["neighbour_entropy"],
        "global_entropy": r["global_entropy"],
        "num_boundaries": r["num_boundaries"]
    } for r in results_list])
    
    df.to_csv(os.path.join(args.output_dir, "entropy_metrics.csv"), index=False)
    
    stats_res = compute_statistics(df)
    
    with open(os.path.join(args.output_dir, "statistical_summary.txt"), "w") as f:
        f.write(json.dumps(stats_res, indent=4))
        
    # 4. Generate Figures
    logging.info("Generating ICASSP figures...")
    generate_figures(results_list, df, args.output_dir)
    
    # 5. Print Final Summary
    print("\n" + "="*50)
    print("FINAL ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total Utterances Analyzed : {stats_res['total_utterances']}")
    print(f"Total Boundaries Detected : {stats_res['total_boundaries']}")
    print(f"Average Boundary Entropy  : {stats_res['avg_boundary']:.4f} bits")
    print(f"Average Neighbour Entropy : {stats_res['avg_neighbour']:.4f} bits")
    print(f"Average Global Entropy    : {stats_res['avg_global']:.4f} bits")
    print("-" * 50)
    print("STATISTICAL SIGNIFICANCE")
    print(f"Paired t-test p-value     : {stats_res['p_val_t']:.4e}")
    print(f"Wilcoxon p-value          : {stats_res['p_val_w']:.4e}")
    print(f"Effect Size (Cohen's d)   : {stats_res['cohens_d']:.4f}")
    print("="*50 + "\n")
    logging.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()

"""
2026-07-31 14:21:11,475 [INFO] Logging initialized. Results will be saved to ./analysis_results
2026-07-31 14:21:11,512 [INFO] Loading XTTS tokenizer from /home/spark2/Models/XTTS-v2/config.json
2026-07-31 14:21:11,517 [INFO] Found 20 utterances. Commencing analysis...
Processing Attentions: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:05<00:00,  3.85it/s]
/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/scipy/stats/_morestats.py:4088: UserWarning: Exact p-value calculation does not work if there are zeros. Switching to normal approximation.
  warnings.warn("Exact p-value calculation does not work if there are "
Traceback (most recent call last):
  File "/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/analysis_pipeline.py", line 447, in <module>
    main()
  File "/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/analysis_pipeline.py", line 423, in main
    f.write(json.dumps(stats_res, indent=4))
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/json/__init__.py", line 238, in dumps
    **kw).encode(obj)
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/json/encoder.py", line 201, in encode
    chunks = list(chunks)
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/json/encoder.py", line 431, in _iterencode
    yield from _iterencode_dict(o, _current_indent_level)
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/json/encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/json/encoder.py", line 438, in _iterencode
    o = _default(o)
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/json/encoder.py", line 179, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type int64 is not JSON serializable
"""
