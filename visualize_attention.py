import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

# ------------------------

ATTN_FILE = "./xtts_attention_results/000_103085_w5Jyq3XMbb3WwiKQ_0000/generation_attentions.pkl"

SAVE_DIR = "./attention_heatmaps"

os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------

with open(ATTN_FILE, "rb") as f:
    attentions = pickle.load(f)

print("Timesteps :", len(attentions))
print("Layers    :", len(attentions[0]))

# Last decoding timestep
last_step = attentions[-1]

# Visualize a few representative layers
layers = [0, 10, 20, 29]

for layer in layers:

    att = last_step[layer]

    # (1, heads, Q, K)
    att = att.squeeze(0)

    # Average across heads
    att = att.mean(dim=0)

    att = att.cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(att, aspect="auto", origin="lower")
    plt.colorbar()
    plt.title(f"Layer {layer}")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            f"layer_{layer}.png"
        ),
        dpi=200
    )

    plt.close()

print("Done.")
