(icassp_cstts) spark2@01HW2722098:~/users/intern/Atreyee-Das/ICASSP_Work/implementation$ python run_ablation.py
=====================================================================================
STARTING REAL ABLATION STUDY: Zero-Shot Code-Switched Speech Synthesis
=====================================================================================
[+] Loading Offline Whisper Model from: /home/spark2/Models/whisper_large_v3_turbo
[✓] Whisper pipeline loaded successfully!
[+] Loading Real XTTS-v2 Model (CPU-First Initialization)...
GPT2InferenceModel has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
  - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
  - If you are not the owner of the model architecture class, please contact the model code owner to update it.
[+] Pushing TTS Engine to GPU...

[Evaluating Arm 1 (Full System)] generating real audio...
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
You have passed task=transcribe, but also have set `forced_decoder_ids` to [[1, None], [2, 50360]] which creates a conflict. `forced_decoder_ids` will be ignored in favor of task=transcribe.
Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.43.0. You should pass an instance of `EncoderDecoderCache` instead, e.g. `past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`.
You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset

[Evaluating Arm 2 (Minus Guardrail)] generating real audio...

[Evaluating Arm 3 (Minus L_entropy)] generating real audio...

[Evaluating Arm 4 (Minus IPA Unification)] generating real audio...

==========================================================================================
Ablation Arm                   | MCD (dB)   | SIM-R    | WER (%)  | Δ Entropy  | RTF   
==========================================================================================
Arm 1 (Full System)            | 70.84      | 0.69     | 56.60    | 0.03       | 0.363 
Arm 2 (Minus Guardrail)        | 72.25      | 0.69     | 52.10    | 0.03       | 0.354 
Arm 3 (Minus L_entropy)        | 71.23      | 0.68     | 49.80    | 0.45       | 0.355 
Arm 4 (Minus IPA Unification)  | 71.68      | 0.70     | 49.20    | 0.04       | 0.353 
==========================================================================================
[✓] Physical audio files saved to: /home/spark2/users/intern/Atreyee-Das/ICASSP_Work/implementation/ablation_outputs
