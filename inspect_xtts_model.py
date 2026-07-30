import os
import torch

from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig


# ============================================================
# CHANGE ONLY THIS PATH IF NEEDED
# ============================================================

MODEL_DIR = "/home/spark2/Models/XTTS-v2"

CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
CHECKPOINT_DIR = MODEL_DIR

print("=" * 80)
print("Loading XTTS model...")
print("=" * 80)

# ------------------------------------------------------------
# Load config
# ------------------------------------------------------------

config = XttsConfig()
config.load_json(CONFIG_PATH)

# ------------------------------------------------------------
# Build model
# ------------------------------------------------------------

model = Xtts.init_from_config(config)

# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

model.load_checkpoint(
    config,
    checkpoint_dir=CHECKPOINT_DIR,
    eval=True
)

model.eval()

print("\n")
print("=" * 80)
print("MODEL LOADED SUCCESSFULLY")
print("=" * 80)

# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nModel class:")
print(type(model))

print("\n")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")

print("\n")
print("=" * 80)
print("TOP LEVEL CHILDREN")
print("=" * 80)

for name, module in model.named_children():
    print(f"{name:35s} --> {type(module)}")

print("\n")
print("=" * 80)
print("FULL MODULE TREE")
print("=" * 80)

for name, module in model.named_modules():
    print(f"{name:80s} {type(module)}")
