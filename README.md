# TCS-Paper1

log terminals -1
================================================================================
STARTING REAL ABLATION STUDY: Zero-Shot Code-Switched Speech Synthesis
================================================================================
[+] Loading Real XTTS-v2 Model (CPU-First Initialization)...
GPT2InferenceModel has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
  - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
  - If you are not the owner of the model architecture class, please contact the model code owner to update it.
[+] Pushing TTS Engine to RTX PRO 5000 GPU...

[Evaluating Arm 1 (Full System)] generating real audio...
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
You have passed task=transcribe, but also have set `forced_decoder_ids` to [[1, None], [2, 50360]] which creates a conflict. `forced_decoder_ids` will be ignored in favor of task=transcribe.
WARNING:phonemizer:words count mismatch on 100.0% of the lines (1/1)
WARNING:phonemizer:words count mismatch on 100.0% of the lines (1/1)

[Evaluating Arm 2 (Minus Guardrail)] generating real audio...
You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
WARNING:phonemizer:words count mismatch on 100.0% of the lines (1/1)
WARNING:phonemizer:words count mismatch on 100.0% of the lines (1/1)

[Evaluating Arm 4 (Minus IPA Unification)] generating real audio...

=====================================================================================
Ablation Arm                   | MCD (dB)   | SIM-R    | WER (%)  | Δ Entropy  | RTF   
=====================================================================================
Arm 1 (Full System)            | 1332.04    | 0.64     | 64.27    | 0.01       | 0.067 
Arm 2 (Minus Guardrail)        | 1433.07    | 0.65     | 51.70    | 0.00       | 0.068 
Arm 4 (Minus IPA Unification)  | 1305.20    | 0.65     | 66.87    | 0.01       | 0.067 
=====================================================================================

log-terminals2-
=====================================================================================
Ablation Arm                   | MCD (dB)   | SIM-R    | WER (%)  | Δ Entropy  | RTF   
=====================================================================================
Arm 1 (Full System)            | 514.50     | 0.65     | 63.78    | 0.00       | 0.070 
Arm 2 (Minus Guardrail)        | 504.89     | 0.65     | 49.35    | 0.01       | 0.067 
Arm 4 (Minus IPA Unification)  | 520.58     | 0.64     | 75.53    | 0.01       | 0.069 

log terminals 3-
=====================================================================================
Ablation Arm                   | MCD (dB)   | SIM-R    | WER (%)  | Δ Entropy  | RTF   
=====================================================================================
Arm 1 (Full System)            | 520.54     | 0.65     | 52.66    | 0.01       | 0.069 
Arm 2 (Minus Guardrail)        | 497.07     | 0.64     | 52.02    | 0.01       | 0.064 
Arm 4 (Minus IPA Unification)  | 512.85     | 0.65     | 52.10    | 0.01       | 0.068 
=====================================================================================

log terminals 4-
================================================================================
STARTING REAL ABLATION STUDY: Zero-Shot Code-Switched Speech Synthesis
================================================================================
[+] Loading Real XTTS-v2 Model (CPU-First Initialization)...
GPT2InferenceModel has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
  - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
  - If you are not the owner of the model architecture class, please contact the model code owner to update it.
[+] Pushing TTS Engine to RTX PRO 5000 GPU...

[Evaluating Arm 1 (Full System)] generating real audio...
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
You have passed task=transcribe, but also have set `forced_decoder_ids` to [[1, None], [2, 50360]] which creates a conflict. `forced_decoder_ids` will be ignored in favor of task=transcribe.
WARNING:phonemizer:words count mismatch on 100.0% of the lines (1/1)
WARNING:phonemizer:words count mismatch on 100.0% of the lines (1/1)

[Evaluating Arm 2 (Minus Guardrail)] generating real audio...
You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
WARNING:phonemizer:words count mismatch on 100.0% of the lines (1/1)
WARNING:phonemizer:words count mismatch on 100.0% of the lines (1/1)

[Evaluating Arm 4 (Minus IPA Unification)] generating real audio...

=====================================================================================
Ablation Arm                   | MCD (dB)   | SIM-R    | WER (%)  | Δ Entropy  | RTF   
=====================================================================================
Arm 1 (Full System)            | 518.37     | 0.65     | 48.13    | 0.00       | 0.071
Arm 2 (Minus Guardrail)        | 497.94     | 0.66     | 64.46    | 0.01       | 0.072 
Arm 4 (Minus IPA Unification)  | 520.79     | 0.64     | 50.74    | 0.01       | 0.066 
=====================================================================================
[✓] Physical audio files saved to: /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/ablation_outputs

terminal logs 5-
=====================================================================================
STARTING REAL ABLATION STUDY: Zero-Shot Code-Switched Speech Synthesis
=====================================================================================
[+] Loading Real XTTS-v2 Model (CPU-First Initialization)...
GPT2InferenceModel has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
  - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
  - If you are not the owner of the model architecture class, please contact the model code owner to update it.
[+] Pushing TTS Engine to RTX PRO 5000 GPU...

[Evaluating Arm 1 (Full System)] generating real audio...
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0000, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0001, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0002, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0003, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0004, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0005, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0006, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0007, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0008, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0009, skipping MCD evaluation for this file.

[Evaluating Arm 2 (Minus Guardrail)] generating real audio...
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0000, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0001, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0002, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0003, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0004, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0005, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0006, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0007, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0008, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0009, skipping MCD evaluation for this file.

[Evaluating Arm 3 (Minus L_entropy)] generating real audio...
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0000, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0001, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0002, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0003, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0004, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0005, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0006, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0007, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0008, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0009, skipping MCD evaluation for this file.

[Evaluating Arm 4 (Minus IPA Unification)] generating real audio...
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0000, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0001, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0002, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0003, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0004, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0005, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0006, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0007, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0008, skipping MCD evaluation for this file.
  [!] Missing ground truth for 103085_w5Jyq3XMbb3WwiKQ_0009, skipping MCD evaluation for this file.

==========================================================================================
Ablation Arm                   | MCD (dB)   | SIM-R    | WER (%)  | Δ Entropy  | RTF   
==========================================================================================
Arm 1 (Full System)            | 0.00       | 0.00     | 0.00     | 0.00       | 0.000 
Arm 2 (Minus Guardrail)        | 0.00       | 0.00     | 0.00     | 0.00       | 0.000 
Arm 3 (Minus L_entropy)        | 0.00       | 0.00     | 0.00     | 0.00       | 0.000 
Arm 4 (Minus IPA Unification)  | 0.00       | 0.00     | 0.00     | 0.00       | 0.000 
==========================================================================================
[✓] Physical audio files saved to: /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/ablation_outputs
