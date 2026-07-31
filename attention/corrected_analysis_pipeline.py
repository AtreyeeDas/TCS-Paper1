#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
corrected_analysis_pipeline.py

A complete, un-averaged, scientifically rigorous analysis of XTTS-v2 GPT cross-attention.
Investigates localized disruptions in specific layers and heads near code-switch boundaries.
Implements Soft Attention-Weighted Alignment and Benjamini-Hochberg FDR correction.
"""

import os
import glob
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import torch
import warnings

# Suppress minor warnings for clean logs
warnings.filterwarnings("ignore", category=RuntimeWarning)

# XTTS specific imports
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer


def setup_logging(output_dir):
    """Initializes console and file logging."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'corrected_analysis.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Logging initialized. Output directory: {output_dir}")


def bh_fdr_correction(p_values):
    """
    Manual Benjamini-Hochberg False Discovery Rate (FDR) correction.
    Returns adjusted p-values.
    """
    p_values = np.asarray(p_values)
    n = len(p_values)
    if n == 0:
        return p_values
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    adjusted_p_values = np.zeros(n)
    min_adj = 1.0
    for i in range(n - 1, -1, -1):
        p = sorted_p_values[i]
        adj_p = (p * n) / (i + 1)
        min_adj = min(min_adj, adj_p)
        adjusted_p_values[sorted_indices[i]] = min_adj
        
    return np.clip(adjusted_p_values, 0, 1)


def load_xtts_tokenizer(config_path):
    """Reloads the XTTS tokenizer."""
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
    """Classifies a decoded token string using Unicode block detection."""
    if not text:
        return "SPACE"
    special_tokens = ["[START]", "[STOP]", "[en]", "[hi]", "[zh]"]
    if text in special_tokens or (text.startswith("[") and text.endswith("]")):
        return "SPECIAL"
    if text.isspace() or text == "_" or text == " ":
        return "SPACE"
    if any('\u0900' <= c <= '\u097F' for c in text): # Devanagari
        return "HINDI"
    if any('a' <= c.lower() <= 'z' for c in text): # Basic Latin
        return "ENGLISH"
    return "OTHER"


def detect_boundaries(token_classes):
    """Detects English <-> Hindi transitions. Returns list of transition indices."""
    boundaries = []
    last_lang = None
    last_lang_idx = -1
    for i, cls in enumerate(token_classes):
        if cls in ["ENGLISH", "HINDI"]:
            if last_lang is not None and cls != last_lang:
                boundaries.append(i) # Tag the exact token where switch occurs
            last_lang = cls
            last_lang_idx = i
    return boundaries


def compute_metrics(A):
    """
    Computes 15 raw metrics for a 4D attention tensor A (T, L, H, K).
    Returns a dictionary of metric arrays, each shaped (T, L, H).
    """
    T, L, H, K = A.shape
    A = np.clip(A, 1e-12, 1.0)
    
    metrics = {}
    
    # 1. Entropy
    metrics['entropy'] = -np.sum(A * np.log2(A), axis=3)
    
    # 2. Peak probability
    metrics['peak_prob'] = np.max(A, axis=3)
    
    # 3. Centre of Mass (CoM)
    k_indices = np.arange(K).reshape(1, 1, 1, K)
    com = np.sum(A * k_indices, axis=3)
    metrics['com'] = com
    
    # 4. Variance
    diff = k_indices - com[..., np.newaxis]
    metrics['variance'] = np.sum(A * (diff ** 2), axis=3)
    
    # 5. Width (keys > threshold, e.g., > 0.05)
    metrics['width'] = np.sum(A > 0.05, axis=3)
    
    # 6. Diagonal deviation (Distance from monotonic progression t/T * K)
    t_indices = np.arange(T).reshape(T, 1, 1)
    expected_com = (t_indices / max(T-1, 1)) * (K - 1)
    metrics['diag_dev'] = np.abs(com - expected_com)
    
    # 7. Jump distance (Delta CoM)
    jump = np.zeros((T, L, H))
    jump[1:] = np.abs(com[1:] - com[:-1])
    metrics['jump_dist'] = jump
    
    # 8. Velocity
    vel = np.zeros((T, L, H))
    vel[1:] = com[1:] - com[:-1]
    metrics['velocity'] = vel
    
    # 9. Acceleration
    accel = np.zeros((T, L, H))
    accel[1:] = vel[1:] - vel[:-1]
    metrics['acceleration'] = accel
    
    # 10. Local KL divergence (A_t || A_{t-1})
    kl = np.zeros((T, L, H))
    A_prev = np.clip(np.roll(A, shift=1, axis=0), 1e-12, 1.0)
    kl[1:] = np.sum(A[1:] * np.log2(A[1:] / A_prev[1:]), axis=3)
    metrics['local_kl'] = kl
    
    # 11. Earth Mover Distance (Wasserstein 1D via CDF difference)
    cdf = np.cumsum(A, axis=3)
    cdf_prev = np.roll(cdf, shift=1, axis=0)
    emd = np.zeros((T, L, H))
    emd[1:] = np.sum(np.abs(cdf[1:] - cdf_prev[1:]), axis=3)
    metrics['emd'] = emd
    
    # 12. Sharpness (Inverse Participation Ratio)
    metrics['sharpness'] = np.sum(A ** 2, axis=3)
    
    # 13. Sparsity (Fraction of near-zero elements)
    metrics['sparsity'] = np.sum(A < 0.01, axis=3) / K
    
    # 14. Concentration (Sum of top 3 probabilities)
    sorted_A = np.sort(A, axis=3)
    metrics['concentration'] = np.sum(sorted_A[..., -3:], axis=3)
    
    # 15. Entropy derivative
    ent_deriv = np.zeros((T, L, H))
    ent_deriv[1:] = metrics['entropy'][1:] - metrics['entropy'][:-1]
    metrics['ent_deriv'] = ent_deriv
    
    return metrics


def soft_alignment(A, metric_tensor):
    """
    Performs Soft Attention-Weighted Alignment.
    A: (T, L, H, K)
    metric_tensor: (T, L, H)
    Returns aligned_metric: (L, H, K)
    """
    # M_aligned(l,h,k) = sum_t (A_t,l,h,k * M_t,l,h) / sum_t (A_t,l,h,k)
    A_sum = np.sum(A, axis=0) # (L, H, K)
    A_sum = np.clip(A_sum, 1e-12, None)
    
    weighted_metric = np.sum(A * metric_tensor[..., np.newaxis], axis=0) # (L, H, K)
    aligned_metric = weighted_metric / A_sum
    return aligned_metric


def process_utterance(utterance_dir, tokenizer):
    """Processes raw layers/heads and performs soft alignment."""
    attention_path = os.path.join(utterance_dir, "generation_attentions.pkl")
    tokens_path = os.path.join(utterance_dir, "text_tokens.pt")
    
    if not os.path.exists(attention_path) or not os.path.exists(tokens_path):
        return None

    text_tokens = torch.load(tokens_path, map_location="cpu").squeeze()
    text_len = text_tokens.shape[0]
    
    token_classes = [classify_token(tokenizer.decode([tid])) for tid in text_tokens.tolist()]
    boundaries = detect_boundaries(token_classes)
    
    if not boundaries:
        return None

    with open(attention_path, "rb") as f:
        attentions = pickle.load(f)
        
    num_gen_steps = len(attentions)
    if num_gen_steps == 0:
        return None
        
    # Isolate text keys for all heads and layers
    # attentions shape over steps: List[Tuple(Layers)] -> layer shape (1, 16, q, k)
    first_step_attn = attentions[0][0]
    audio_cond_len = first_step_attn.shape[-1] - text_len
    
    L = len(attentions[0])
    H = first_step_attn.shape[1]
    
    # Build 4D Tensor: (T, L, H, K)
    A = np.zeros((num_gen_steps, L, H, text_len))
    for t in range(num_gen_steps):
        for l in range(L):
            last_q_attn = attentions[t][l][0, :, -1, :] # (H, K_total)
            text_keys_attn = last_q_attn[:, audio_cond_len : audio_cond_len + text_len]
            # Normalize over text keys to isolate focus on text
            row_sums = text_keys_attn.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            A[t, l, :, :] = text_keys_attn / row_sums

    # Compute raw metrics (T, L, H)
    raw_metrics_dict = compute_metrics(A)
    
    # Soft align metrics to text tokens (L, H, K)
    aligned_metrics = {}
    for m_name, m_tensor in raw_metrics_dict.items():
        aligned_metrics[m_name] = soft_alignment(A, m_tensor)
        
    return {
        "utt_id": os.path.basename(os.path.normpath(utterance_dir)),
        "boundaries": boundaries,
        "aligned_metrics": aligned_metrics, # Dict of (L, H, K) arrays
        "text_len": text_len,
        "A": A # For trajectory plotting
    }


def analyze_boundaries(results_list, window_sizes=[1, 2, 3, 5, 8]):
    """Extracts boundary, neighbourhood, and global stats for every layer/head/metric."""
    # Data structure to hold: metric -> L -> H -> region -> list of values
    # Regions: 'boundary', 'global', 'window_±1', etc.
    analysis = defaultdict(lambda: np.zeros((30, 16), dtype=object))
    
    metrics = results_list[0]['aligned_metrics'].keys()
    
    # Initialize nested lists
    for m in metrics:
        for l in range(30):
            for h in range(16):
                analysis[m][l, h] = defaultdict(list)
                
    for res in results_list:
        bounds = res['boundaries']
        text_len = res['text_len']
        
        for m in metrics:
            metric_data = res['aligned_metrics'][m] # (L, H, K)
            
            for l in range(30):
                for h in range(16):
                    head_data = metric_data[l, h, :]
                    
                    # Global
                    valid_global = head_data[~np.isnan(head_data)]
                    if len(valid_global) > 0:
                        analysis[m][l, h]['global'].append(np.mean(valid_global))
                    
                    # Boundary
                    b_vals = [head_data[b] for b in bounds if not np.isnan(head_data[b])]
                    if b_vals:
                        analysis[m][l, h]['boundary'].append(np.mean(b_vals))
                        
                    # Windows
                    for w in window_sizes:
                        w_vals = []
                        for b in bounds:
                            start = max(0, b - w)
                            end = min(text_len, b + w + 1)
                            for i in range(start, end):
                                if i != b and not np.isnan(head_data[i]):
                                    w_vals.append(head_data[i])
                        if w_vals:
                            analysis[m][l, h][f'window_{w}'].append(np.mean(w_vals))
                            
    return analysis


def perform_statistical_testing(analysis_data, compare_against='global'):
    """Performs T-test, Wilcoxon, Cohen's d, and FDR correction per metric."""
    stat_results = {}
    
    for m, matrix in analysis_data.items():
        p_vals_t = np.ones((30, 16))
        p_vals_w = np.ones((30, 16))
        effect_sizes = np.zeros((30, 16))
        
        flat_p_t = []
        indices = []
        
        for l in range(30):
            for h in range(16):
                b_dist = np.array(matrix[l, h]['boundary'])
                comp_dist = np.array(matrix[l, h][compare_against])
                
                # We need paired lengths. If utts dropped due to nans, take minimum
                min_len = min(len(b_dist), len(comp_dist))
                if min_len > 3: # Need minimum N for valid stats
                    b_dist = b_dist[:min_len]
                    comp_dist = comp_dist[:min_len]
                    
                    try:
                        _, pt = stats.ttest_rel(b_dist, comp_dist)
                        _, pw = stats.wilcoxon(b_dist, comp_dist)
                        diff = b_dist - comp_dist
                        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-9)
                    except:
                        pt, pw, d = 1.0, 1.0, 0.0
                        
                    p_vals_t[l, h] = pt
                    p_vals_w[l, h] = pw
                    effect_sizes[l, h] = d
                    
                    flat_p_t.append(pt)
                    indices.append((l, h))
                    
        # FDR Correction
        adj_p = bh_fdr_correction(flat_p_t)
        adj_p_matrix = np.ones((30, 16))
        for (l, h), p in zip(indices, adj_p):
            adj_p_matrix[l, h] = p
            
        stat_results[m] = {
            'p_t': p_vals_t,
            'p_w': p_vals_w,
            'p_adj': adj_p_matrix,
            'cohens_d': effect_sizes
        }
        
    return stat_results


def generate_head_heatmaps(stat_results, output_dir):
    """Generates Layer x Head heatmaps for all metrics."""
    os.makedirs(os.path.join(output_dir, 'heatmaps'), exist_ok=True)
    
    sig_summary = []
    
    for m, res in stat_results.items():
        # Effect Size Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(res['cohens_d'], cmap="coolwarm", center=0, annot=False)
        plt.title(f"{m.capitalize()}: Cohen's d (Layer x Head)")
        plt.xlabel("Head Index")
        plt.ylabel("Layer Index")
        plt.savefig(os.path.join(output_dir, 'heatmaps', f"{m}_effect_size.png"), dpi=300)
        plt.close()
        
        # Adjusted P-Value Heatmap (-log10 scale for visibility)
        plt.figure(figsize=(12, 8))
        log_p = -np.log10(np.clip(res['p_adj'], 1e-10, 1.0))
        sns.heatmap(log_p, cmap="Reds", vmin=0, vmax=3) # vmax 3 = p 0.001
        plt.title(f"{m.capitalize()}: -log10(FDR Adjusted P-Value)")
        plt.xlabel("Head Index")
        plt.ylabel("Layer Index")
        plt.savefig(os.path.join(output_dir, 'heatmaps', f"{m}_p_adj.png"), dpi=300)
        plt.close()
        
        # Find significant heads
        sig_indices = np.where(res['p_adj'] < 0.05)
        for l, h in zip(sig_indices[0], sig_indices[1]):
            sig_summary.append({
                'metric': m, 'layer': l, 'head': h,
                'p_adj': res['p_adj'][l, h], 'cohens_d': res['cohens_d'][l, h]
            })
            
    df_sig = pd.DataFrame(sig_summary)
    if not df_sig.empty:
        df_sig = df_sig.sort_values(by=['p_adj', 'cohens_d'])
        df_sig.to_csv(os.path.join(output_dir, 'significant_heads.csv'), index=False)
        
    return df_sig


def automate_conclusion(df_sig):
    """Outputs A, B, or C strictly based on FDR-corrected stats."""
    print("\n" + "="*60)
    print("AUTOMATED SCIENTIFIC CONCLUSION")
    print("="*60)
    
    if df_sig.empty:
        print("\nCONCLUSION C: No evidence that decoder attention changes around language-switch boundaries.")
        print("Reason: 0 heads passed Benjamini-Hochberg FDR correction (p < 0.05).")
        print("No statistically reliable localized decoder attention effect was found.")
    else:
        strong_evidence = df_sig[(df_sig['p_adj'] < 0.01) & (df_sig['cohens_d'].abs() > 0.5)]
        if not strong_evidence.empty:
            print("\nCONCLUSION A: Strong evidence that specific decoder heads behave differently near language-switch boundaries.")
            print(f"Reason: {len(strong_evidence)} head/metric combinations passed strict significance (FDR p < 0.01, |d| > 0.5).")
            print("Top disrupting heads:")
            print(strong_evidence.head(5).to_string(index=False))
        else:
            print("\nCONCLUSION B: Weak evidence requiring larger datasets.")
            print(f"Reason: {len(df_sig)} heads passed FDR p < 0.05, but effect sizes were marginal (|d| < 0.5) or p-values were borderline.")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./xtts_attention_results")
    parser.add_argument("--config_path", type=str, default="/home/spark2/Models/XTTS-v2/config.json")
    parser.add_argument("--output_dir", type=str, default="./corrected_analysis_results")
    args = parser.parse_args()

    setup_logging(args.output_dir)
    tokenizer = load_xtts_tokenizer(args.config_path)
    
    utterance_dirs = [d for d in glob.glob(os.path.join(args.data_dir, "*_*")) if os.path.isdir(d)]
    if not utterance_dirs:
        logging.error("No data found.")
        return
        
    logging.info("Extracting un-averaged T x L x H x K matrices and mapping Soft Alignments...")
    results_list = []
    for u_dir in utterance_dirs:
        res = process_utterance(u_dir, tokenizer)
        if res:
            results_list.append(res)
            
    if not results_list:
        logging.error("No valid boundaries detected.")
        return
        
    logging.info("Segmenting alignments into Boundary vs. Neighbourhood (±1, 2, 3, 5, 8)...")
    analysis_data = analyze_boundaries(results_list)
    
    logging.info("Executing 480-hypothesis FDR corrected statistical testing...")
    stat_results = perform_statistical_testing(analysis_data, compare_against='window_3') # Test against ±3 window
    
    logging.info("Generating Layer x Head localization heatmaps...")
    df_sig = generate_head_heatmaps(stat_results, args.output_dir)
    
    automate_conclusion(df_sig)


if __name__ == "__main__":
    main()
"""
. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import resource_filename
2026-07-31 15:14:54,467 [INFO] Logging initialized. Output directory: ./corrected_analysis_results
2026-07-31 15:14:54,468 [INFO] Loading XTTS tokenizer from /home/spark2/Models/XTTS-v2/config.json
2026-07-31 15:14:54,473 [INFO] Extracting un-averaged T x L x H x K matrices and mapping Soft Alignments...
Traceback (most recent call last):
  File "/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/corrected_analysis_pipeline.py", line 476, in <module>
    main()
  File "/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/corrected_analysis_pipeline.py", line 455, in main
    res = process_utterance(u_dir, tokenizer)
  File "/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/corrected_analysis_pipeline.py", line 249, in process_utterance
    A[t, l, :, :] = text_keys_attn / row_sums
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/_tensor.py", line 1257, in __array__
    return self.numpy().astype(dtype, copy=False)
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
"""
