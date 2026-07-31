import pickle
import torch
import matplotlib.pyplot as plt
import os

ATTN_FILE = "./xtts_attention_results/000_103085_w5Jyq3XMbb3WwiKQ_0000/generation_attentions.pkl"

with open(ATTN_FILE, "rb") as f:
    attentions = pickle.load(f)

layer = 29          # try 0,10,20,29 later
head_average = True

rows = []

for step in attentions:

    # step[layer] -> (1, heads, query_len, key_len)
    A = step[layer].squeeze(0)

    if head_average:
        A = A.mean(0)

    # keep ONLY newest query
    last_query = A[-1]

    rows.append(last_query)

# pad rows to equal length
max_len = max(r.shape[0] for r in rows)

matrix = torch.zeros(len(rows), max_len)

for i, r in enumerate(rows):
    matrix[i, : len(r)] = r

plt.figure(figsize=(10,8))
plt.imshow(matrix.numpy(), aspect="auto", origin="lower")
plt.colorbar()

plt.xlabel("Key Position")
plt.ylabel("Generation Step")
plt.title(f"Layer {layer}")

os.makedirs("analysis", exist_ok=True)
plt.savefig("analysis/layer29_generation_matrix.png", dpi=250)

plt.show()
