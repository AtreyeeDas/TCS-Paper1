#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
corrected_analysis_pipeline.py

A complete, un-averaged, scientifically rigorous analysis of XTTS-v2 GPT cross-attention.
Investigates localized disruptions in specific layers and heads near code-switch boundaries.
Implements Soft Attention-Weighted Alignment and Benjamini-Hochberg FDR correction.

EXTENDED MODULES ADDED:
- Module 1: Functional Head Discovery
- Module 2: Functional Head Ablation
- Module 3: Behaviour Clustering
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

# ML clustering & dimensionality reduction imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

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
        
    first_step_attn = attentions[0][0]
    audio_cond_len = first_step_attn.shape[-1] - text_len
    
    L = len(attentions[0])
    H = first_step_attn.shape[1]
    
    A = np.zeros((num_gen_steps, L, H, text_len))
    for t in range(num_gen_steps):
        for l in range(L):
            last_q_attn = attentions[t][l][0, :, -1, :] # (H, K_total)
            text_keys_attn = last_q_attn[:, audio_cond_len : audio_cond_len + text_len]
            text_keys_attn = text_keys_attn.float().cpu().numpy()
            
            row_sums = text_keys_attn.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            A[t, l, :, :] = text_keys_attn / row_sums

    raw_metrics_dict = compute_metrics(A)
    
    aligned_metrics = {}
    for m_name, m_tensor in raw_metrics_dict.items():
        aligned_metrics[m_name] = soft_alignment(A, m_tensor)
        
    return {
        "utt_id": os.path.basename(os.path.normpath(utterance_dir)),
        "boundaries": boundaries,
        "aligned_metrics": aligned_metrics, 
        "text_len": text_len,
        "A": A,
        "token_classes": token_classes # EXTENSION ADDITION
    }


def analyze_boundaries(results_list, window_sizes=[1, 2, 3, 5, 8]):
    """Extracts boundary, neighbourhood, and global stats for every layer/head/metric."""
    analysis = defaultdict(lambda: np.zeros((30, 16), dtype=object))
    metrics = results_list[0]['aligned_metrics'].keys()
    
    for m in metrics:
        for l in range(30):
            for h in range(16):
                analysis[m][l, h] = defaultdict(list)
                
    for res in results_list:
        bounds = res['boundaries']
        text_len = res['text_len']
        
        for m in metrics:
            metric_data = res['aligned_metrics'][m] 
            
            for l in range(30):
                for h in range(16):
                    head_data = metric_data[l, h, :]
                    
                    valid_global = head_data[~np.isnan(head_data)]
                    if len(valid_global) > 0:
                        analysis[m][l, h]['global'].append(np.mean(valid_global))
                    
                    b_vals = [head_data[b] for b in bounds if not np.isnan(head_data[b])]
                    if b_vals:
                        analysis[m][l, h]['boundary'].append(np.mean(b_vals))
                        
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
                
                min_len = min(len(b_dist), len(comp_dist))
                if min_len > 3: 
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
        plt.figure(figsize=(12, 8))
        sns.heatmap(res['cohens_d'], cmap="coolwarm", center=0, annot=False)
        plt.title(f"{m.capitalize()}: Cohen's d (Layer x Head)")
        plt.xlabel("Head Index")
        plt.ylabel("Layer Index")
        plt.savefig(os.path.join(output_dir, 'heatmaps', f"{m}_effect_size.png"), dpi=300)
        plt.close()
        
        plt.figure(figsize=(12, 8))
        log_p = -np.log10(np.clip(res['p_adj'], 1e-10, 1.0))
        sns.heatmap(log_p, cmap="Reds", vmin=0, vmax=3) 
        plt.title(f"{m.capitalize()}: -log10(FDR Adjusted P-Value)")
        plt.xlabel("Head Index")
        plt.ylabel("Layer Index")
        plt.savefig(os.path.join(output_dir, 'heatmaps', f"{m}_p_adj.png"), dpi=300)
        plt.close()
        
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
    if df_sig.empty:
        return "C", "No evidence that decoder attention changes around language-switch boundaries."
    else:
        strong = df_sig[(df_sig['p_adj'] < 0.01) & (df_sig['cohens_d'].abs() > 0.5)]
        if not strong.empty:
            return "A", "Strong evidence that specific decoder heads behave differently near language-switch boundaries."
        else:
            return "B", "Weak evidence requiring larger datasets (significant but small effect sizes)."

# =============================================================================
# MODULE 1: FUNCTIONAL HEAD DISCOVERY
# =============================================================================
def module_1_functional_discovery(results_list, df_sig, output_dir):
    logging.info("Running Module 1: Functional Head Discovery...")
    os.makedirs(output_dir, exist_ok=True)
    
    # If no heads are significant, fall back to analyzing all heads to output exploratory data
    target_heads = df_sig[['layer', 'head']].drop_duplicates().values if not df_sig.empty else [(l,h) for l in range(30) for h in range(16)]
    
    head_stats = []
    
    for (l, h) in target_heads:
        l, h = int(l), int(h)
        metrics_accum = defaultdict(list)
        
        for res in results_list:
            A = res['A'] # (T, L, H, K)
            tokens = res['token_classes']
            bounds = res['boundaries']
            K = res['text_len']
            T = A.shape[0]
            
            A_head = A[:, l, h, :] # (T, K)
            token_mass = np.sum(A_head, axis=0) # (K,)
            total_mass = np.sum(token_mass) + 1e-9
            
            # Linguistic Ratios
            eng_mass = sum(token_mass[i] for i, cls in enumerate(tokens) if cls == 'ENGLISH')
            hin_mass = sum(token_mass[i] for i, cls in enumerate(tokens) if cls == 'HINDI')
            
            # Boundary Ratio (±2 window)
            bound_indices = set()
            for b in bounds:
                bound_indices.update(range(max(0, b-2), min(K, b+3)))
            bound_mass = sum(token_mass[i] for i in bound_indices)
            
            non_bound_mass = total_mass - bound_mass
            expected_bound_mass = total_mass * (len(bound_indices) / K) if K > 0 else 1
            selectivity = bound_mass / (expected_bound_mass + 1e-9)
            
            # Span & Monotonicity
            span = np.sum(token_mass > 0.05)
            step_argmax = np.argmax(A_head, axis=1)
            monotonicity = np.corrcoef(np.arange(T), step_argmax)[0, 1] if T > 1 else 0
            
            metrics_accum['eng'].append(eng_mass/total_mass)
            metrics_accum['hin'].append(hin_mass/total_mass)
            metrics_accum['bound'].append(bound_mass/total_mass)
            metrics_accum['non_bound'].append(non_bound_mass/total_mass)
            metrics_accum['selectivity'].append(selectivity)
            metrics_accum['span'].append(span)
            metrics_accum['monotonicity'].append(monotonicity if not np.isnan(monotonicity) else 0)

        # Average over all utterances
        avg_eng = np.mean(metrics_accum['eng'])
        avg_hin = np.mean(metrics_accum['hin'])
        avg_bound = np.mean(metrics_accum['bound'])
        avg_sel = np.mean(metrics_accum['selectivity'])
        
        # Determine Preferred Class
        pref = 'Boundary' if avg_sel > 1.5 else ('English' if avg_eng > avg_hin else 'Hindi')
        if avg_eng < 0.2 and avg_hin < 0.2 and avg_sel < 1.0: pref = 'Other'
            
        head_stats.append({
            'Layer': l, 'Head': h,
            'EnglishRatio': avg_eng,
            'HindiRatio': avg_hin,
            'BoundaryRatio': avg_bound,
            'Selectivity': avg_sel,
            'AverageSpan': np.mean(metrics_accum['span']),
            'Monotonicity': np.mean(metrics_accum['monotonicity']),
            'PreferredClass': pref,
            'SpecializationScore': avg_sel * (avg_eng + avg_hin)
        })
        
    df_head = pd.DataFrame(head_stats)
    df_head = df_head.sort_values(by='Selectivity', ascending=False)
    df_head.to_csv(os.path.join(output_dir, 'head_specialization.csv'), index=False)
    
    # Generate Heatmaps
    for metric in ['Selectivity', 'EnglishRatio', 'HindiRatio']:
        matrix = np.zeros((30, 16))
        for _, row in df_head.iterrows():
            matrix[int(row['Layer']), int(row['Head'])] = row[metric]
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix, cmap="viridis")
        plt.title(f"Module 1: {metric} Map")
        plt.xlabel("Head"); plt.ylabel("Layer")
        plt.savefig(os.path.join(output_dir, f"{metric.lower()}_heatmap.png"), dpi=300)
        plt.close()
        
    return df_head

# =============================================================================
# MODULE 2: FUNCTIONAL HEAD ABLATION
# =============================================================================
def module_2_head_ablation(results_list, df_sig, output_dir):
    logging.info("Running Module 2: Functional Head Ablation...")
    
    target_heads = df_sig[['layer', 'head']].drop_duplicates().values if not df_sig.empty else [(l,h) for l in range(30) for h in range(16)]
    ablation_stats = []
    
    for (l, h) in target_heads:
        l, h = int(l), int(h)
        metrics_drift = defaultdict(list)
        
        for res in results_list:
            A = res['A']
            K = res['text_len']
            T = A.shape[0]
            
            # Original state
            A_orig = A[:, l, h, :]
            com_orig = np.sum(A_orig * np.arange(K), axis=1)
            ent_orig = -np.sum(A_orig * np.log2(np.clip(A_orig, 1e-12, 1.0)), axis=1)
            
            # Ablated state (Mean of other heads in layer)
            A_ablated = np.mean(np.delete(A[:, l, :, :], h, axis=1), axis=1)
            com_abl = np.sum(A_ablated * np.arange(K), axis=1)
            ent_abl = -np.sum(A_ablated * np.log2(np.clip(A_ablated, 1e-12, 1.0)), axis=1)
            
            # Drifts
            align_drift = np.mean(np.abs(com_orig - com_abl))
            ent_change = np.mean(ent_abl - ent_orig)
            
            A_prev_orig = np.clip(np.roll(A_orig, shift=1, axis=0), 1e-12, 1.0)
            kl_orig = np.sum(A_orig[1:] * np.log2(np.clip(A_orig[1:], 1e-12, 1.0) / A_prev_orig[1:]), axis=1)
            
            A_prev_abl = np.clip(np.roll(A_ablated, shift=1, axis=0), 1e-12, 1.0)
            kl_abl = np.sum(A_ablated[1:] * np.log2(np.clip(A_ablated[1:], 1e-12, 1.0) / A_prev_abl[1:]), axis=1)
            
            kl_increase = np.mean(kl_abl - kl_orig)
            
            metrics_drift['align_drift'].append(align_drift)
            metrics_drift['ent_change'].append(ent_change)
            metrics_drift['kl_increase'].append(kl_increase)
            
        ablation_stats.append({
            'Layer': l, 'Head': h,
            'AlignmentDrift': np.mean(metrics_drift['align_drift']),
            'EntropyChange': np.mean(metrics_drift['ent_change']),
            'KLIncrease': np.mean(metrics_drift['kl_increase'])
        })
        
    df_ablation = pd.DataFrame(ablation_stats)
    df_ablation = df_ablation.sort_values(by='AlignmentDrift', ascending=False)
    df_ablation.to_csv(os.path.join(output_dir, 'head_ablation.csv'), index=False)
    
    matrix = np.zeros((30, 16))
    for _, row in df_ablation.iterrows():
        matrix[int(row['Layer']), int(row['Head'])] = row['AlignmentDrift']
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, cmap="magma")
    plt.title("Module 2: Ablation Importance (Alignment Drift)")
    plt.xlabel("Head"); plt.ylabel("Layer")
    plt.savefig(os.path.join(output_dir, 'ablation_importance_heatmap.png'), dpi=300)
    plt.close()
    
    return df_ablation

# =============================================================================
# MODULE 3: BEHAVIOUR CLUSTERING
# =============================================================================
def module_3_behaviour_clustering(analysis_data, df_mod1, output_dir):
    logging.info("Running Module 3: Behaviour Clustering...")
    
    # Construct Feature Matrix for 480 heads
    features = ['entropy', 'kl', 'emd', 'jump_dist', 'velocity', 'sharpness', 'width', 'diag_dev']
    
    X = []
    labels = []
    
    for l in range(30):
        for h in range(16):
            vec = []
            for f in features:
                # Use global average across all utterances if it exists in analysis_data
                if f in analysis_data:
                    val = np.mean(analysis_data[f][l, h]['global'])
                elif f == 'kl': val = np.mean(analysis_data['local_kl'][l, h]['global'])
                else: val = 0.0
                vec.append(val)
                
            # Bring in Selectivity from Mod1 if available
            mod1_row = df_mod1[(df_mod1['Layer'] == l) & (df_mod1['Head'] == h)]
            vec.append(mod1_row['Selectivity'].values[0] if not mod1_row.empty else 0.0)
            
            X.append(vec)
            labels.append((l, h))
            
    X = np.nan_to_num(np.array(X))
    X_scaled = StandardScaler().fit_transform(X)
    
    # Optimal Cluster Selection via Silhouette
    best_k, best_score = 2, -1
    for k in range(2, min(10, len(X))):
        preds = KMeans(n_clusters=k, random_state=42).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, preds)
        if score > best_score:
            best_score = score
            best_k = k
            
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # PCA & TSNE
    pca = PCA(n_components=2).fit_transform(X_scaled)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)
    
    df_clusters = pd.DataFrame(X, columns=features + ['Selectivity'])
    df_clusters['Layer'] = [x[0] for x in labels]
    df_clusters['Head'] = [x[1] for x in labels]
    df_clusters['Cluster'] = clusters
    df_clusters['PCA1'] = pca[:, 0]; df_clusters['PCA2'] = pca[:, 1]
    df_clusters['TSNE1'] = tsne[:, 0]; df_clusters['TSNE2'] = tsne[:, 1]
    
    # Automatically Describe Clusters
    cluster_summaries = {}
    for c in range(best_k):
        c_data = df_clusters[df_clusters['Cluster'] == c]
        name = "Generic Heads"
        if c_data['Selectivity'].mean() > df_clusters['Selectivity'].mean() + c_data['Selectivity'].std():
            name = "Boundary Heads"
        elif c_data['jump_dist'].mean() > df_clusters['jump_dist'].mean():
            name = "Transition Heads"
        elif c_data['diag_dev'].mean() < df_clusters['diag_dev'].mean():
            name = "Sharp Monotonic Heads"
        elif c_data['width'].mean() > df_clusters['width'].mean():
            name = "Diffuse Heads"
            
        cluster_summaries[c] = {
            'Name': name,
            'Count': len(c_data),
            'AvgEntropy': c_data['entropy'].mean(),
            'AvgSelectivity': c_data['Selectivity'].mean()
        }
        
    df_clusters['ClusterName'] = df_clusters['Cluster'].map(lambda x: cluster_summaries[x]['Name'])
    df_clusters.to_csv(os.path.join(output_dir, 'head_clusters.csv'), index=False)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='ClusterName', data=df_clusters, palette='Set1')
    plt.title("Module 3: PCA of Head Behaviours")
    plt.savefig(os.path.join(output_dir, 'pca_heads.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='ClusterName', data=df_clusters, palette='Set2')
    plt.title("Module 3: t-SNE of Head Behaviours")
    plt.savefig(os.path.join(output_dir, 'tsne_heads.png'), dpi=300)
    plt.close()
    
    matrix = np.zeros((30, 16))
    for _, row in df_clusters.iterrows():
        matrix[int(row['Layer']), int(row['Head'])] = row['Cluster']
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, cmap="tab10", annot=False)
    plt.title("Module 3: Head Cluster Distribution")
    plt.xlabel("Head"); plt.ylabel("Layer")
    plt.savefig(os.path.join(output_dir, 'cluster_heatmap.png'), dpi=300)
    plt.close()
    
    return cluster_summaries

# =============================================================================
# MAIN EXECUTOR & FINAL REPORT
# =============================================================================
def generate_final_report(df_sig, df_mod1, df_mod2, cluster_summaries, class_conclusion):
    print("\n" + "="*50)
    print("FINAL RESEARCH REPORT")
    print("="*50)
    print(f"Number of significant heads (FDR corrected) : {len(df_sig) if not df_sig.empty else 0}")
    
    print("\nTop Specialized Heads (Module 1):")
    if not df_mod1.empty:
        print(df_mod1[['Layer', 'Head', 'Selectivity', 'PreferredClass']].head(5).to_string(index=False))
        
    print("\nMost Important Ablation Heads (Module 2):")
    if not df_mod2.empty:
        print(df_mod2[['Layer', 'Head', 'AlignmentDrift', 'EntropyChange']].head(5).to_string(index=False))
        
    print(f"\nNumber of Clusters (Module 3): {len(cluster_summaries)}")
    print("Cluster Summaries:")
    for k, v in cluster_summaries.items():
        print(f"  - Cluster {k} ({v['Name']}): N={v['Count']}, Avg Entropy={v['AvgEntropy']:.2f}, Avg Selectivity={v['AvgSelectivity']:.2f}")
        
    print("\nINTERPRETATION:")
    
    # Dynamic interpretation based on extracted metrics
    has_specialization = df_mod1['Selectivity'].max() > 1.5 if not df_mod1.empty else False
    has_boundary_heads = any("Boundary" in v['Name'] for v in cluster_summaries.values())
    global_instability = class_conclusion == "C" # From FDR automated conclusion
    
    print(f"Does evidence support functional specialization? {'YES' if has_specialization else 'NO'}")
    print(f"Does evidence support boundary-specialized heads? {'YES' if has_boundary_heads else 'NO'}")
    print(f"Does evidence support global decoder instability? {'YES' if global_instability else 'NO'}")
    print("="*50 + "\n")


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
    stat_results = perform_statistical_testing(analysis_data, compare_against='window_3') 
    
    logging.info("Generating Layer x Head localization heatmaps...")
    df_sig = generate_head_heatmaps(stat_results, args.output_dir)
    class_code, class_msg = automate_conclusion(df_sig)
    
    # Execute New Modules
    df_mod1 = module_1_functional_discovery(results_list, df_sig, args.output_dir)
    df_mod2 = module_2_head_ablation(results_list, df_sig, args.output_dir)
    cluster_summaries = module_3_behaviour_clustering(analysis_data, df_mod1, args.output_dir)
    
    # Generate Final Compiled Report
    generate_final_report(df_sig, df_mod1, df_mod2, cluster_summaries, class_code)

if __name__ == "__main__":
    main()
