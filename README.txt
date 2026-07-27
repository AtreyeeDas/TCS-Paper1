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


malized   : nauttheviyU pratti eka salIDiyU meM nauttha joDane kI suvidhA detA hai, jo kI prastutI karaNa ke vakta najara nahIM thA|

  [Diagnostics] Ground Truth : handout view slides को हैन्डाउट के रूप में print करने की सुविधा देता है
  [Diagnostics] Raw Whisper  : हंदोत वियूस राइड़क को हैंडावड के रूप में प्रींट करने की सुविधा देता है
  [Diagnostics] Normalized   : haMdota viyUsa rAi.Daka ko haiMDAvaDa ke rUpa meM prIMTa karane kI suvidhA detA hai

  [Diagnostics] Ground Truth : यहाँ हमें एक पेज पर कितने slides print करने हैं इसका चुनाव कर सकते हैं
  [Diagnostics] Raw Whisper  : यहां हमें एक पेज पर कितने स्लीद प्रींट करने हैं, इसका चुनाओ कर सकते हैं.
  [Diagnostics] Normalized   : yahAM hameM eka peja para kitane slIda prIMTa karane haiM, isakA chunAo kara sakate haiM.

  [Diagnostics] Ground Truth : slide sorter view slides की थम्बनेल बताता है
  [Diagnostics] Raw Whisper  : स्लाइड़िद्या, सौर्टई
  [Diagnostics] Normalized   : slAi.DidyA, saurTaI

  [Diagnostics] Ground Truth : अब फिर से normal view button पर click करते हैं
  [Diagnostics] Raw Whisper  : आप भारी गीरा मारी ओवाई थेगी, सहाच से नर्मे दो आउर।
  [Diagnostics] Normalized   : Apa bhArI gIrA mArI ovAI thegI, sahAcha se narme do Aura|

  [Diagnostics] Ground Truth : screen की बाईं ओर आप slides pane देखते हैं यह प्रस्तुति में slides के थम्बनेल सम्मिलित करता है
  [Diagnostics] Raw Whisper  : स्क्रीम की बाई और आप स्लीधत पाने देखते हैं, यह प्रस्तुती में स्लीधत के थम ने समिलित करता है.
  [Diagnostics] Normalized   : skrIma kI bAI aura Apa slIdhata pAne dekhate haiM, yaha prastutI meM slIdhata ke thama ne samilita karatA hai.

  [Diagnostics] Ground Truth : दाएँ तरफ tasks pane हैं जिसमें अनुभाग है
  [Diagnostics] Raw Whisper  : दाई तरफ तक्स पाने हैं जिसमें अनुभाग है।
  [Diagnostics] Normalized   : dAI tarapha taksa pAne haiM jisameM anubhAga hai|

  [Diagnostics] Ground Truth : लेआउट्स section में पहले से ही कुछ सैम्पल लेआउट्स मौजूद हैं
  [Diagnostics] Raw Whisper  : Layouts section में पहले से ही कुछ sample layouts मौझूद हैं.
  [Diagnostics] Normalized   : Layouts section meM pahale se hI kuCha sample layouts maujhUda haiM.

  [Diagnostics] Ground Truth : हम उनका उपयोग ऐसे ही कर सकते हैं या आवश्यकता अनुसार कुछ बदलाव करके उपयोग कर सकते हैं
  [Diagnostics] Raw Whisper  : हम उनका उपयोग ऐसे ही कर सकते हैं या आवश्यक्ता अनुसार कुछ बदलाव करके उपयोग कर सकते हैं.
  [Diagnostics] Normalized   : hama unakA upayoga aise hI kara sakate haiM yA AvashyaktA anusAra kuCha badalAva karake upayoga kara sakate haiM.

  [Diagnostics] Ground Truth : जैसे जैसे हम इन tutorial में आगे बढ़ेंगे इन sections को विस्तार में देखेंगे
  [Diagnostics] Raw Whisper  : जैसे जैसे हम इन तुटोरियाल में आगे बढ़ेंगे इन सेक्शोंज को विस्तार में देखेंगे.
  [Diagnostics] Normalized   : jaise jaise hama ina tuToriyAla meM Age ba.DheMge ina sekshoMja ko vistAra meM dekheMge.

==========================================================================================
Ablation Arm                   | MCD (dB)   | SIM-R    | WER (%)  | Δ Entropy  | RTF   
==========================================================================================
Arm 1 (Full System)            | 2564.49    | 0.69     | 100.02   | 0.02       | 0.349 
Arm 2 (Minus Guardrail)        | 2510.54    | 0.69     | 102.19   | 0.02       | 0.359 
Arm 3 (Minus L_entropy)        | 2514.45    | 0.67     | 103.16   | 0.45       | 0.355 
Arm 4 (Minus IPA Unification)  | 2506.78    | 0.67     | 102.59   | 0.03       | 0.355 
==========================================================================================
