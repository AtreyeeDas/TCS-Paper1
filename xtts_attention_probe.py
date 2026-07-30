"""
xtts_attention_probe.py

Purpose:
--------
Determine whether XTTS's internal GPT2Model can expose
decoder self-attention matrices.

This script DOES NOT synthesize speech.

It only loads XTTS and probes the decoder.
"""

import torch
import numpy as np

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


#############################################
# CHANGE THESE
#############################################

CONFIG_PATH = "/path/to/config.json"
CHECKPOINT_DIR = "/path/to/XTTS-v2"

#############################################


device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*80)
print("Loading XTTS...")
print("="*80)

config = XttsConfig()
config.load_json(CONFIG_PATH)

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_dir=CHECKPOINT_DIR,
    eval=True
)

model.to(device)
model.eval()

print("\nLoaded.\n")

############################################################
# Access GPT
############################################################

gpt_wrapper = model.gpt
gpt_model = gpt_wrapper.gpt

print("="*80)
print("GPT MODEL")
print("="*80)

print(type(gpt_model))

print("\nCONFIG")
print(gpt_model.config)

print("\nNumber of layers:",
      len(gpt_model.h))

print("\nAttention implementation:")

if hasattr(gpt_model.config, "_attn_implementation"):
    print(gpt_model.config._attn_implementation)
else:
    print("No attribute")

############################################################
# Dummy tokens
############################################################

batch = 1
seq = 16

hidden = gpt_model.config.n_embd

dummy_embeds = torch.randn(
    batch,
    seq,
    hidden,
    device=device
)

############################################################
# Forward pass
############################################################

print("\n")
print("="*80)
print("Attempting output_attentions=True")
print("="*80)

try:

    outputs = gpt_model(
        inputs_embeds=dummy_embeds,
        output_attentions=True,
        return_dict=True
    )

    print("\nForward successful.")

    if hasattr(outputs, "attentions"):
        print("outputs.attentions exists")

        if outputs.attentions is None:
            print("BUT it is None.")

        else:

            print("Returned",
                  len(outputs.attentions),
                  "layers.")

            for i, attn in enumerate(outputs.attentions):

                if attn is None:
                    print(f"Layer {i}: None")

                else:
                    print(
                        f"Layer {i}:",
                        tuple(attn.shape)
                    )

            np.save(
                "sample_attention.npy",
                outputs.attentions[0].detach().cpu().numpy()
            )

            print("\nSaved sample_attention.npy")

    else:
        print("No outputs.attentions field.")

except Exception as e:

    print("\nFAILED")
    print(type(e))
    print(e)


"""
If attentions are NOT returned
Run this immediately afterward.
print("\n")
print("="*80)
print("GPT BLOCK TYPES")
print("="*80)

for i, block in enumerate(gpt_model.h):

    print(i)

    print(type(block))

    print(type(block.attn))

    print(block.attn)

    print("-"*50)
If you see
GPT2SdpaAttention
then SDPA is swallowing the attention matrices.
Then run this
print(gpt_model.config)

print(vars(gpt_model.config))
Look for
_attn_implementation
If it prints
sdpa
that's exactly what I expect.
If it says "sdpa"
Before the forward pass, insert
gpt_model.config._attn_implementation = "eager"
Then rerun
outputs = gpt_model(
    inputs_embeds=dummy_embeds,
    output_attentions=True,
    return_dict=True,
)
There is a good chance this alone will restore real attention tensors, because Hugging Face GPT-2 supports multiple attention implementations and the eager implementation can return attention probabilities.
If that still returns None
Then the fallback is to replace each GPT2SdpaAttention module with the eager GPT2Attention implementation (or monkey-patch the SDPA call) while keeping the pretrained weights. Since your uploaded model structure confirms XTTS uses a standard Hugging Face GPT-2 backbone, this is feasible.
"""
