import os
import torch

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# --------------------------------------------------
# Your XTTS-v2 local model directory
# --------------------------------------------------

MODEL_DIR = "/home/spark2/Models/XTTS-v2"

CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "model.pth")

# XTTS repositories usually contain:
# vocab.json
# speakers_xtts.pth
# etc.

# --------------------------------------------------
# Device
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("Device:", device)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 80)

# --------------------------------------------------
# Load config
# --------------------------------------------------

config = XttsConfig()
config.load_json(CONFIG_PATH)

print("Config loaded.")

# --------------------------------------------------
# Load model
# --------------------------------------------------

model = Xtts.init_from_config(config)

model.load_checkpoint(
    config,
    checkpoint_path=CHECKPOINT_PATH,
    eval=True,
)

model.to(device)

print("\nModel successfully loaded.\n")

print("=" * 80)
print("MODEL TYPE")
print(type(model))
print("=" * 80)

# --------------------------------------------------
# Print immediate children
# --------------------------------------------------

print("\nTOP LEVEL MODULES\n")

for name, module in model.named_children():
    print(f"{name:40s} --> {type(module)}")

print("=" * 80)

# --------------------------------------------------
# Print ALL modules
# --------------------------------------------------

print("\nFULL MODULE TREE\n")

for idx, (name, module) in enumerate(model.named_modules()):
    print(f"{idx:05d} | {name:80s} | {type(module)}")

print("=" * 80)

# --------------------------------------------------
# Search for attention-like modules
# --------------------------------------------------

print("\nATTENTION RELATED MODULES\n")

keywords = [
    "att",
    "attention",
    "cross",
    "self",
    "multi",
    "gpt",
    "transformer",
]

for name, module in model.named_modules():

    lname = name.lower()

    if any(k in lname for k in keywords):

        print(f"{name:80s} | {type(module)}")

print("=" * 80)

# --------------------------------------------------
# Parameter count
# --------------------------------------------------

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nPARAMETERS")

print(f"Total:      {total:,}")
print(f"Trainable:  {trainable:,}")

print("=" * 80)

"""
import TTS
import torch

print("TTS version :", TTS.__version__)
print("Torch       :", torch.__version__)
print("CUDA        :", torch.version.cuda) 
"""
