import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import os

MODEL_DIR="/home/spark2/Models/XTTS-v2"

config=XttsConfig()
config.load_json(os.path.join(MODEL_DIR,"config.json"))

model=Xtts.init_from_config(config)

model.load_checkpoint(
    config,
    checkpoint_dir=MODEL_DIR,
    eval=True,
)

token_file="./xtts_attention_results/000_103085_w5Jyq3XMbb3WwiKQ_0000/text_tokens.pt"

tokens=torch.load(token_file)

print("Shape:",tokens.shape)

tokenizer=model.tokenizer

ids=tokens.squeeze().tolist()

print("\nTOKEN IDS\n")
print(ids)

print("\nDECODED TOKENS\n")

for i,t in enumerate(ids):

    try:
        piece=tokenizer.decode([t])
    except:
        piece=str(t)

    print(f"{i:03d} : {repr(piece)}")
