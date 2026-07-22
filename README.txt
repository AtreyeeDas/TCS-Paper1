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
(icassp_cstts) spark2@01HW2722098:~/users/intern/Atreyee-Das/ICASSP_Work/implementation$ python mucs_slicer.py
[+] Found 3136 utterances to slice. Loading audio files...
Processing Recordings:   0%|                                                                                                                                            | 0/30 [00:00<?, ?it/s]/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/librosa/core/intervals.py:8: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import resource_filename
/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/mucs_slicer.py:55: UserWarning: PySoundFile failed. Trying audioread instead.
  y, sr = librosa.load(audio_path, sr=24000)
/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
        Deprecated as of librosa version 0.10.0.
        It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)
[-] Failed to process recording w5Jyq3XMbb3WwiKQ: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/w5Jyq3XMbb3WwiKQ.wav'
Processing Recordings:   3%|████▍                                                                                                                               | 1/30 [00:00<00:13,  2.23it/s][-] Failed to process recording UO5ZLDtkc8QMi9Hk: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/UO5ZLDtkc8QMi9Hk.wav'
[-] Failed to process recording nJfFvNUB8tQkaGgp: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/nJfFvNUB8tQkaGgp.wav'
[-] Failed to process recording x8iijxWeCf26KNpC: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/x8iijxWeCf26KNpC.wav'
[-] Failed to process recording swLUjPHatcTk0wf8: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/swLUjPHatcTk0wf8.wav'
[-] Failed to process recording fpL6MBSZsci9dyAn: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/fpL6MBSZsci9dyAn.wav'
[-] Failed to process recording TkzR0GrQyK0vSTuj: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/TkzR0GrQyK0vSTuj.wav'
[-] Failed to process recording ZQC2TFqLsvNDq9bP: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/ZQC2TFqLsvNDq9bP.wav'
[-] Failed to process recording FV8hkymGWcy53lrf: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/FV8hkymGWcy53lrf.wav'
[-] Failed to process recording NhV290ew2hDJfVqc: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/NhV290ew2hDJfVqc.wav'
[-] Failed to process recording HgEJsLGKNWSFl8T7: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/HgEJsLGKNWSFl8T7.wav'
[-] Failed to process recording K5BlLYB9ynemLTVa: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/K5BlLYB9ynemLTVa.wav'
[-] Failed to process recording 4oLp3bc9OSJbDrwM: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/4oLp3bc9OSJbDrwM.wav'
[-] Failed to process recording LPGZ2Jo8uBX4sx76: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/LPGZ2Jo8uBX4sx76.wav'
[-] Failed to process recording 5baFeyjOxK8JQnNG: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/5baFeyjOxK8JQnNG.wav'
[-] Failed to process recording n3lwFhZkSdbJkOa0: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/n3lwFhZkSdbJkOa0.wav'
[-] Failed to process recording aVTioioyNUVPGF4t: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/aVTioioyNUVPGF4t.wav'
[-] Failed to process recording RRrEfkw4Rh5TdJkS: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/RRrEfkw4Rh5TdJkS.wav'
[-] Failed to process recording Psh5UDSFbhWLZ3gk: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/Psh5UDSFbhWLZ3gk.wav'
[-] Failed to process recording 406yMKxIdSDHRf8H: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/406yMKxIdSDHRf8H.wav'
[-] Failed to process recording nImMxiSAAmCMR3yi: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/nImMxiSAAmCMR3yi.wav'
[-] Failed to process recording K8YJLqIcMPJ5GftE: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/K8YJLqIcMPJ5GftE.wav'
[-] Failed to process recording VU6OA7HUqTpQNtm2: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/VU6OA7HUqTpQNtm2.wav'
[-] Failed to process recording dNHXkqqVPqWYhzis: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/dNHXkqqVPqWYhzis.wav'
[-] Failed to process recording kGQKLyQ1HtnUNL1d: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/kGQKLyQ1HtnUNL1d.wav'
[-] Failed to process recording lHGLSlYIb1clRBRF: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/lHGLSlYIb1clRBRF.wav'
[-] Failed to process recording Bmcrju1ByfdrXE9P: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/Bmcrju1ByfdrXE9P.wav'
[-] Failed to process recording jFTbEEy3BCO5axj5: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/jFTbEEy3BCO5axj5.wav'
[-] Failed to process recording kvxzesc5pGxL6dnk: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/kvxzesc5pGxL6dnk.wav'
[-] Failed to process recording 4xyIm2P6Xzlin341: [Errno 2] No such file or directory: '/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts/4xyIm2P6Xzlin341.wav'
Processing Recordings: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 66.60it/s]
[✓] Slicing complete! All files saved to ./MUCS_sliced/audio

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


Terminal Logs 6-

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
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
You have passed task=transcribe, but also have set `forced_decoder_ids` to [[1, None], [2, 50360]] which creates a conflict. `forced_decoder_ids` will be ignored in favor of task=transcribe.
WARNING:phonemizer:words count mismatch on 100.0% of the lines (1/1)
WARNING:phonemizer:words count mismatch on 100.0% of the lines (1/1)

[Evaluating Arm 2 (Minus Guardrail)] generating real audio...
You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
WARNING:phonemizer:words count mismatch on 100.0% of the lines (1/1)
WARNING:phonemizer:words count mismatch on 100.0% of the lines (1/1)

[Evaluating Arm 3 (Minus L_entropy)] generating real audio...
WARNING:phonemizer:words count mismatch on 100.0% of the lines (1/1)
WARNING:phonemizer:words count mismatch on 100.0% of the lines (1/1)

[Evaluating Arm 4 (Minus IPA Unification)] generating real audio...

==========================================================================================

Ablation Arm                   | MCD (dB)   | SIM-R    | WER (%)  | Δ Entropy  | RTF   

==========================================================================================

Arm 1 (Full System)            | 712.32     | 0.65     | 51.73    | 0.00       | 0.130 
Arm 2 (Minus Guardrail)        | 689.32     | 0.65     | 50.62    | 0.01       | 0.126 
Arm 3 (Minus L_entropy)        | 692.87     | 0.64     | 50.78    | 0.50       | 0.126 
Arm 4 (Minus IPA Unification)  | 700.27     | 0.64     | 48.00    | 0.01       | 0.125 

==========================================================================================

[✓] Physical audio files saved to: /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/ablation_outputs

==========================================================================================
[✓] Physical audio files saved to: /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/ablation_outputs


[+] Initializing Training on cuda (Blackwell / RTX PRO 5000 Optimized)
[!] SpeechBrain fallback initialized due to: module 'torchaudio' has no attribute 'list_audio_backends'
[!] SpeechBrain fallback initialized due to: module 'torchaudio' has no attribute 'list_audio_backends'
[+] Starting Fine-Tuning & Validation Loop...
/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/librosa/core/intervals.py:8: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import resource_filename
Traceback (most recent call last):
  File "/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/train.py", line 112, in <module>
    train_entropy_regularized_model()
  File "/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/train.py", line 47, in train_entropy_regularized_model
    for batch in train_loader:
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 725, in __next__
    data = self._next_data()
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 785, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 57, in fetch
    return self.collate_fn(data)
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/utils/data/_utils/collate.py", line 401, in default_collate
    return collate(batch, collate_fn_map=default_collate_fn_map)
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/utils/data/_utils/collate.py", line 171, in collate
    {
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/utils/data/_utils/collate.py", line 172, in <dictcomp>
    key: collate(
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/utils/data/_utils/collate.py", line 155, in collate
    return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/utils/data/_utils/collate.py", line 275, in collate_tensor_fn
    return torch.stack(batch, 0, out=out)

[+] Initializing Training on cuda (Blackwell / RTX PRO 5000 Optimized)
/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/speechbrain/utils/autocast.py:68: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  wrapped_fwd = torch.cuda.amp.custom_fwd(fwd, cast_inputs=cast_inputs)
[+] Starting Fine-Tuning & Validation Loop...
/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/librosa/core/intervals.py:8: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import resource_filename
Traceback (most recent call last):
  File "/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/train.py", line 173, in <module>
    train_entropy_regularized_model()
  File "/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/train.py", line 112, in train_entropy_regularized_model
    predicted_audio, cross_attn_matrix = model(ipa_tokens, speaker_emb)
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1780, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1791, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/src/model_architecture.py", line 48, in forward
    decoder_out, cross_attn_matrix = self.cross_attention_layer(
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1780, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1791, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/src/stage3_4_engine.py", line 32, in forward
    attn_out, _ = self.self_attn(x, x, x, is_causal=True)
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1780, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1791, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/nn/modules/activation.py", line 1495, in forward
    attn_output, attn_output_weights = F.multi_head_attention_forward(
  File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/nn/functional.py", line 6837, in multi_head_attention_forward
    raise RuntimeError(
RuntimeError: Need attn_mask if specifying the is_causal hint. You may use the Transformer module method `generate_square_subsequent_mask` to create this mask.

what caused this error- [+] Initializing Training on cuda (Blackwell / RTX PRO 5000 Optimized) /home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/speechbrain/utils/autocast.py:68: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.   wrapped_fwd = torch.cuda.amp.custom_fwd(fwd, cast_inputs=cast_inputs) [+] Starting Fine-Tuning & Validation Loop... /home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/librosa/core/intervals.py:8: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.   from pkg_resources import resource_filename /__w/pytorch/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:111: operator(): block: [0,0,0], thread: [0,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed. Traceback (most recent call last):   File "/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/train.py", line 173, in <module>     train_entropy_regularized_model()   File "/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/train.py", line 118, in train_entropy_regularized_model     torch.stack([   File "/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/train.py", line 119, in <listcomp>     entropy_criterion(   File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1780, in _wrapped_call_impl     return self._call_impl(*args, **kwargs)   File "/home/spark2/miniconda3/envs/icassp_cstts/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1791, in _call_impl     return forward_call(*args, **kwargs)   File "/home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/src/stage3_4_engine.py", line 96, in forward     boundary_entropy = entropy_per_frame[:, valid_boundaries] torch.AcceleratorError: CUDA error: device-side assert triggered Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information. CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. For debugging consider passing CUDA_LAUNCH_BLOCKING=1
