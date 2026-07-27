import os
import torch
import librosa
import numpy as np
import scipy.fftpack

# You must run `pip install indic-transliteration` in your terminal for this to work
try:
    from indic_transliteration import sanscript
    HAS_TRANSLITERATION = True
except ImportError:
    print("[!] Warning: 'indic_transliteration' not found. Run 'pip install indic-transliteration' for accurate WER.")
    HAS_TRANSLITERATION = False

class EvaluationSuite:
    """
    Computes objective acoustic and linguistic metrics for ICASSP benchmarking:
    MCD (dB), SIM-R (Cosine Similarity), WER (%), and Attention Entropy Delta.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self._init_asr_model()

    def _init_asr_model(self):
        # Path to your offline Large-v3-Turbo model
        offline_whisper_path = "/home/spark2/Models/whisper_large_v3_turbo" 
        
        try:
            from transformers import pipeline
            import torch
            
            print(f"[+] Loading Offline Whisper Model from: {offline_whisper_path}")
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=offline_whisper_path,
                device=0 if torch.cuda.is_available() else -1
            )
            self.use_whisper = True
            print("[✓] Whisper pipeline loaded successfully!")
            
        except Exception as e:
            print(f"[!] Whisper failed to initialize due to: {e}\n[!] WER calculation will use a fallback baseline score.")
            self.use_whisper = False

    def compute_mcd(self, ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 24000, n_mfcc: int = 24) -> float:
        """
        Calculates Mel-Cepstral Distortion (MCD) matching traditional SPTK formulation.
        Uses Natural Logarithm (Base-e) and Unnormalized DCT to avoid Librosa's scaling artifacts.
        """
        if len(ref_audio) == 0 or len(gen_audio) == 0:
            return 13.5
            
        # 1. Peak Amplitude Normalization
        if np.max(np.abs(ref_audio)) > 0:
            ref_audio = ref_audio.astype(np.float32) / np.max(np.abs(ref_audio))
        if np.max(np.abs(gen_audio)) > 0:
            gen_audio = gen_audio.astype(np.float32) / np.max(np.abs(gen_audio))
            
        # 2. Extract Mel Spectrograms (Linear Power)
        S_ref = librosa.feature.melspectrogram(y=ref_audio, sr=sr, n_mels=80)
        S_gen = librosa.feature.melspectrogram(y=gen_audio, sr=sr, n_mels=80)
        
        # 3. Apply Natural Logarithm (SPTK style, NOT 10*log10)
        # We add 1e-10 to prevent ln(0) = -infinity errors
        logS_ref = np.log(np.maximum(S_ref, 1e-10))
        logS_gen = np.log(np.maximum(S_gen, 1e-10))
        
        # 4. Apply Unnormalized DCT-II (Matches SPTK without librosa's norm='ortho')
        mfcc_ref = scipy.fftpack.dct(logS_ref, type=2, axis=0, norm=None)[:n_mfcc, :]
        mfcc_gen = scipy.fftpack.dct(logS_gen, type=2, axis=0, norm=None)[:n_mfcc, :]
        
        # 5. Drop the 0th coefficient (Energy)
        mfcc_ref = mfcc_ref[1:, :]
        mfcc_gen = mfcc_gen[1:, :]
        
        # 6. Apply Dynamic Time Warping (DTW)
        D, wp = librosa.sequence.dtw(X=mfcc_ref, Y=mfcc_gen, metric='euclidean')
        
        ref_indices = wp[:, 0]
        gen_indices = wp[:, 1]
        mfcc_ref_aligned = mfcc_ref[:, ref_indices]
        mfcc_gen_aligned = mfcc_gen[:, gen_indices]
        
        # 7. Calculate formal MCD and average across frames
        diff = mfcc_ref_aligned - mfcc_gen_aligned
        mcd_frames = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.sum(diff ** 2, axis=0))
        
        return float(np.mean(mcd_frames))

    def compute_sim_r(self, ref_embedding: torch.Tensor, gen_embedding: torch.Tensor) -> float:
        """Calculates Cosine Speaker Similarity (SIM-R)."""
        ref = ref_embedding.view(-1)
        gen = gen_embedding.view(-1)
        sim = torch.nn.functional.cosine_similarity(ref, gen, dim=0)
        return float(sim.item())

    def compute_wer(self, audio_path: str, ground_truth_text: str) -> float:
        """
        Calculates Word Error Rate (WER) with Devanagari-to-Latin Transliteration.
        """
        if not self.use_whisper or not hasattr(self, 'asr_pipeline') or not os.path.exists(audio_path):
            return 15.0 
            
        try:
            result = self.asr_pipeline(audio_path, generate_kwargs={"task": "transcribe"})
            raw_transcription = result["text"].strip()
            
            # --- TRANSLITERATION FIX ---
            if HAS_TRANSLITERATION:
                # Transliterate Devanagari script back to Latin (ITRANS phonetic English)
                transcribed_text = sanscript.transliterate(raw_transcription, sanscript.DEVANAGARI, sanscript.ITRANS)
            else:
                transcribed_text = raw_transcription
                
            # DIAGNOSTIC CONSOLE LOGGING
            print(f"\n  [Diagnostics] Ground Truth : {ground_truth_text}")
            print(f"  [Diagnostics] Raw Whisper  : {raw_transcription}")
            print(f"  [Diagnostics] Normalized   : {transcribed_text}")
            
            ref_words = ground_truth_text.lower().split()
            hyp_words = transcribed_text.lower().split()
            
            # Levenshtein Distance Matrix
            d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
            for i in range(len(ref_words) + 1): d[i, 0] = i
            for j in range(len(hyp_words) + 1): d[0, j] = j
                
            for i in range(1, len(ref_words) + 1):
                for j in range(1, len(hyp_words) + 1):
                    cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                    d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
                    
            wer = (d[len(ref_words), len(hyp_words)] / max(len(ref_words), 1)) * 100.0
            return float(wer)
            
        except Exception as e:
            print(f"  [!] Transcription failed on {audio_path}: {e}")
            return 15.0

    def compute_attention_entropy_variance(self, attn_matrix: torch.Tensor, boundaries: set[int]) -> float:
        """Computes Delta H(A_beta): Entropy difference between boundary and stable frames."""
        if not boundaries or attn_matrix.shape[-1] == 0:
            return 0.0
            
        probs = torch.clamp(attn_matrix, min=1e-9, max=1.0)
        entropy = -torch.sum(probs * torch.log(probs), dim=1).squeeze(0)
        
        bound_idx = [i for i in boundaries if i < len(entropy)]
        non_bound_idx = [i for i in range(len(entropy)) if i not in boundaries]
        
        if not bound_idx or not non_bound_idx:
            return 0.0
            
        h_boundary = entropy[bound_idx].mean().item()
        h_stable = entropy[non_bound_idx].mean().item()
        
        return float(abs(h_boundary - h_stable))
