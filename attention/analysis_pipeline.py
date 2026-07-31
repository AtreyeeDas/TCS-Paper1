import os
import re
import json
import pickle
import logging
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm
from TTS.tts.models.xtts import XttsTokenizer

# ==========================================
# CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_tokenizer(config_path):
    """Reloads the exact XTTS BPE tokenizer used during generation."""
    logging.info(f"Loading XTTS tokenizer from {config_path}...")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # Using Coqui's native tokenizer init if available, 
        # or fallback to standard HF if path provided is direct vocab
        tokenizer = XttsTokenizer(config_path)
        return tokenizer
    except Exception as e:
        logging.warning(f"Native XttsTokenizer load failed: {e}. Attempting HF fallback...")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(os.path.dirname(config_path))

def classify_token(token_str):
    """Classifies token into ENGLISH, HINDI, SPECIAL, SPACE, or OTHER."""
    token_str = token_str.strip()
    if not token_str:
        return "SPACE"
    if re.match(r'^\[[a-zA-Z0-9_]+\]$', token_str):
        return "SPECIAL"
    if re.search(r'[\u0900-\u097F]', token_str):
        return "HINDI"
    if re.search(r'[a-zA-Z]', token_str):
        return "ENGLISH"
    return "OTHER"

def compute_shannon_entropy(prob_dist):
    """Computes Shannon entropy of a probability distribution."""
    prob_dist = prob_dist / (np.sum(prob_dist) + 1e-12)
    return -np.sum(prob_dist * np.log2(prob_dist + 1e-12))

def cohens_d(x, y):
    """Computes Cohen's d effect size for two paired arrays."""
    diff = x - y
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)

# ==========================================
# CORE ANALYSIS PIPELINE
# ==========================================
def analyze_utterance(utterance_dir, tokenizer, window_size=5):
    """Processes a single utterance folder to extract boundary/non-boundary entropy."""
    attentions_path = os.path.join(utterance_dir, "generation_attentions.pkl")
    tokens_path = os.path.join(utterance_dir, "text_tokens.pt")
    
    if not (os.path.exists(attentions_path) and os.path.exists(tokens_path)):
        return None
        
    with open(attentions_path, 'rb') as f:
        attentions = pickle.load(f)
    
    text_tokens = torch.load(tokens_path, map_location='cpu').squeeze().tolist()
    if not isinstance(text_tokens, list):
        text_tokens = [text_tokens]
        
    num_text_tokens = len(text_tokens)
    
    # 1. Decode and Classify Tokens
    decoded_tokens = [tokenizer.decode([t]) for t in text_tokens]
    token_classes = [classify_token(t) for t in decoded_tokens]
    
    # 2. Detect Language Boundaries (English <-> Hindi)
    boundaries = []
    for i in range(1, num_text_tokens):
        prev_cls = token_classes[i-1]
        curr_cls = token_classes[i]
        valid_langs = {"ENGLISH", "HINDI"}
        if prev_cls in valid_langs and curr_cls in valid_langs and prev_cls != curr_cls:
            boundaries.append(i)
            
    # 3. Reconstruct GenerationStep x KeyPosition matrix (Averaged across heads/layers)
    num_gen_steps = len(attentions)
    attn_matrix = np.zeros((num_gen_steps, num_text_tokens))
    
    for t_step in range(num_gen_steps):
        step_attns = attentions[t_step] 
        num_layers = len(step_attns)
        layer_avg = 0
        for layer_idx in range(num_layers):
            # Shape: (batch, heads, q_len, k_len). We want the last query attending to text keys.
            attn = step_attns[layer_idx][0].cpu().numpy() 
            q_attn = attn[:, -1, :num_text_tokens] 
            layer_avg += np.mean(q_attn, axis=0)
        
        layer_avg /= num_layers
        # Normalize to create a valid probability distribution over text tokens
        layer_avg /= (np.sum(layer_avg) + 1e-12)
        attn_matrix[t_step, :] = layer_avg

    # 4. Compute Entropy per Generation Step
    step_entropies = np.array([compute_shannon_entropy(attn_matrix[t, :]) for t in range(num_gen_steps)])
    
    # 5. Map Generation Steps to Text Tokens (via argmax attention)
    aligned_text_indices = np.argmax(attn_matrix, axis=1)
    
    boundary_entropies = []
    non_boundary_entropies = []
    
    for t_step in range(num_gen_steps):
        focused_token = aligned_text_indices[t_step]
        
        # Check if the focused token is within the window of any boundary
        is_boundary = False
        for b_idx in boundaries:
            if abs(focused_token - b_idx) <= window_size:
                is_boundary = True
                break
                
        if is_boundary:
            boundary_entropies.append(step_entropies[t_step])
        else:
            non_boundary_entropies.append(step_entropies[t_step])
            
    return {
        "utterance": os.path.basename(utterance_dir),
        "attn_matrix": attn_matrix,
        "step_entropies": step_entropies,
        "boundaries": boundaries,
        "aligned_text_indices": aligned_text_indices,
        "avg_boundary_entropy": np.mean(boundary_entropies) if boundary_entropies else np.nan,
        "avg_non_boundary_entropy": np.mean(non_boundary_entropies) if non_boundary_entropies else np.nan,
        "num_boundaries": len(boundaries)
    }

# ==========================================
# VISUALIZATION & EXPORT
# ==========================================
def generate_figures(results, output_dir):
    logging.info("Generating analytical figures...")
    
    # Extract clean data for plotting
    clean_results = [r for r in results if not np.isnan(r['avg_boundary_entropy']) and not np.isnan(r['avg_non_boundary_entropy'])]
    b_ents = [r['avg_boundary_entropy'] for r in clean_results]
    nb_ents = [r['avg_non_boundary_entropy'] for r in clean_results]
    
    # D. Boundary vs Non-boundary boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[b_ents, nb_ents], palette="Set2")
    plt.xticks([0, 1], ["Boundary\n(±5 tokens)", "Non-Boundary"])
    plt.ylabel("Shannon Entropy (bits)")
    plt.title("Cross-Attention Entropy Distribution")
    plt.savefig(os.path.join(output_dir, "entropy_boxplot.png"), dpi=300)
    plt.close()
    
    # E. Histogram of entropy difference
    plt.figure(figsize=(8, 6))
    diffs = np.array(b_ents) - np.array(nb_ents)
    sns.histplot(diffs, bins=30, kde=True, color="purple")
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel("Entropy Difference (Boundary - Non-Boundary)")
    plt.ylabel("Frequency (Utterances)")
    plt.title("Distribution of Entropy Differentials")
    plt.savefig(os.path.join(output_dir, "entropy_difference_hist.png"), dpi=300)
    plt.close()
    
    # A & B. Heatmap and Entropy Curve (Using the first utterance with a boundary as an example)
    example_res = next((r for r in clean_results if r['num_boundaries'] > 0), None)
    if example_res:
        # Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(example_res['attn_matrix'].T, cmap="viridis", cbar_kws={'label': 'Attention Weight'})
        for b_idx in example_res['boundaries']:
            plt.axhline(b_idx, color='red', linestyle='--', alpha=0.7, label="Code-Switch Boundary")
        plt.xlabel("Generation Step")
        plt.ylabel("Text Token Index")
        plt.title(f"Decoder Attention Mapping: {example_res['utterance']}")
        plt.savefig(os.path.join(output_dir, f"heatmap_{example_res['utterance']}.png"), dpi=300)
        plt.close()
        
        # Entropy curve
        plt.figure(figsize=(12, 4))
        plt.plot(example_res['step_entropies'], label="Step Entropy", color='blue')
        
        # Highlight boundary regions
        for t_step, focus in enumerate(example_res['aligned_text_indices']):
            if any(abs(focus - b) <= 5 for b in example_res['boundaries']):
                plt.scatter(t_step, example_res['step_entropies'][t_step], color='red', s=10)
                
        plt.xlabel("Generation Step")
        plt.ylabel("Entropy")
        plt.title(f"Autoregressive Entropy Trajectory: {example_res['utterance']}")
        plt.savefig(os.path.join(output_dir, f"entropy_curve_{example_res['utterance']}.png"), dpi=300)
        plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Analyze Code-Switched XTTS Attention Entropy")
    parser.add_argument("--results_dir", type=str, default="xtts_attention_results", help="Path to XTTS outputs")
    parser.add_argument("--config_path", type=str, default="/home/spark2/Models/XTTS-v2/config.json", help="XTTS config")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory for final analysis assets")
    parser.add_argument("--window", type=int, default=5, help="Token window around boundary to classify as localized")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = load_tokenizer(args.config_path)
    
    utterances = [os.path.join(args.results_dir, d) for d in os.listdir(args.results_dir) if os.path.isdir(os.path.join(args.results_dir, d))]
    utterances.sort()
    
    results = []
    logging.info(f"Starting analysis on {len(utterances)} utterances...")
    
    for utt_dir in tqdm(utterances, desc="Processing Utterances"):
        res = analyze_utterance(utt_dir, tokenizer, window_size=args.window)
        if res is not None:
            results.append(res)
            
    # Filter out samples lacking boundary data
    valid_results = [r for r in results if not np.isnan(r['avg_boundary_entropy'])]
    
    if not valid_results:
        logging.error("No valid boundaries detected in the dataset. Terminating analysis.")
        return

    # Extract paired metrics
    b_ent = np.array([r['avg_boundary_entropy'] for r in valid_results])
    nb_ent = np.array([r['avg_non_boundary_entropy'] for r in valid_results])
    
    # Statistical Testing
    t_stat, p_val_t = stats.ttest_rel(b_ent, nb_ent)
    w_stat, p_val_w = stats.wilcoxon(b_ent, nb_ent)
    effect_size = cohens_d(b_ent, nb_ent)
    
    # Aggregated Summary
    avg_b = np.mean(b_ent)
    avg_nb = np.mean(nb_ent)
    total_boundaries = sum(r['num_boundaries'] for r in valid_results)
    
    summary = (
        "====================================================\n"
        "   XTTS CODE-SWITCH ATTENTION ENTROPY ANALYSIS      \n"
        "====================================================\n"
        f"Analyzed Utterances       : {len(valid_results)}\n"
        f"Total Detected Boundaries : {total_boundaries}\n"
        "----------------------------------------------------\n"
        f"Average Boundary Entropy  : {avg_b:.4f} bits\n"
        f"Average Non-Bound Entropy : {avg_nb:.4f} bits\n"
        f"Mean Entropy Difference   : {(avg_b - avg_nb):.4f} bits\n"
        "----------------------------------------------------\n"
        f"Paired t-test (p-value)   : {p_val_t:.4e}\n"
        f"Wilcoxon Rank (p-value)   : {p_val_w:.4e}\n"
        f"Effect Size (Cohen's d)   : {effect_size:.4f}\n"
        "====================================================\n"
    )
    
    # Export Text Summary
    print(summary)
    with open(os.path.join(args.output_dir, "statistical_summary.txt"), "w") as f:
        f.write(summary)
        
    # Export CSV Data
    import csv
    csv_path = os.path.join(args.output_dir, "utterance_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Utterance", "Num_Boundaries", "Avg_Boundary_Entropy", "Avg_NonBoundary_Entropy"])
        for r in valid_results:
            writer.writerow([r['utterance'], r['num_boundaries'], r['avg_boundary_entropy'], r['avg_non_boundary_entropy']])
            
    # Generate requested figures
    generate_figures(valid_results, args.output_dir)
    logging.info(f"[✓] Analysis complete. All assets saved to ./{args.output_dir}/")

if __name__ == "__main__":
    main()
