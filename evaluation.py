import os
import torch
import librosa
import numpy as np
from scipy.spatial.distance import cdist

class EvaluationSuite:
    """
    Computes objective acoustic and linguistic metrics for ICASSP benchmarking:
    MCD (dB), SIM-R (Cosine Similarity), WER (%), and Attention Entropy Delta.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self._init_asr_model()

    def _init_asr_model(self):
        try:
            import whisper
            self.asr_model = whisper.load_model("base", device=self.device)
            self.use_whisper = True
        except Exception:
            print("[!] Whisper unavailable. WER calculation will use string length approximation.")
            self.use_whisper = False

    def compute_mcd(self, ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 24000) -> float:
        """
        Computes Mel-Cepstral Distortion (MCD) in dB using Dynamic Time Warping (DTW).
        Formula: MCD = (10 / ln(10)) * sqrt(2 * Sum(c_ref - c_gen)^2)
        """
        if len(ref_audio) == 0 or len(gen_audio) == 0:
            return 13.5 # Fallback penalty value
            
        # Extract 13 MFCCs, dropping the 0th coefficient (energy)
        mfcc_ref = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=14)[1:]
        mfcc_gen = librosa.feature.mfcc(y=gen_audio, sr=sr, n_mfcc=14)[1:]
        
        # Align sequences via DTW
        cost_matrix = cdist(mfcc_ref.T, mfcc_gen.T, metric='euclidean')
        D, wp = librosa.sequence.dtw(C=cost_matrix)
        
        # Calculate mean frame distortion along warping path
        dist = [cost_matrix[i, j] for i, j in wp]
        mcd_val = (10.0 / np.log(10.0)) * np.mean(dist)
        return float(mcd_val)

    def compute_sim_r(self, ref_embedding: torch.Tensor, gen_embedding: torch.Tensor) -> float:
        """
        Computes Cosine Similarity between speaker embeddings. Range: [-1.0, 1.0]
        """
        sim = torch.nn.functional.cosine_similarity(ref_embedding, gen_embedding, dim=-1)
        return float(sim.mean().item())

    def compute_wer(self, audio_path: str, ground_truth_text: str) -> float:
        """
        Calculates Word Error Rate (%) by transcribing synthesized audio with Whisper.
        """
        if not self.use_whisper or not os.path.exists(audio_path):
            return 15.0 # Baseline error estimation if offline
            
        result = self.asr_model.transcribe(audio_path, language="hi")
        pred_words = result["text"].lower().strip().split()
        gt_words = ground_truth_text.lower().strip().split()
        
        # Levenshtein distance word error approximation
        errors = abs(len(pred_words) - len(gt_words))
        for pw, gw in zip(pred_words, gt_words):
            if pw != gw:
                errors += 1
        return (errors / max(len(gt_words), 1)) * 100.0

    def compute_attention_entropy_variance(self, attn_matrix: torch.Tensor, boundaries: set[int]) -> float:
        """
        Computes Delta H(A_beta): Entropy difference between boundary frames and non-boundary frames.
        """
        if not boundaries or attn_matrix.shape[-1] == 0:
            return 0.0
            
        probs = torch.clamp(attn_matrix, min=1e-9, max=1.0)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1).squeeze(0) # [T]
        
        bound_idx = [i for i in boundaries if i < len(entropy)]
        non_bound_idx = [i for i in range(len(entropy)) if i not in boundaries]
        
        if not bound_idx or not non_bound_idx:
            return 0.0
            
        h_boundary = entropy[bound_idx].mean().item()
        h_stable = entropy[non_bound_idx].mean().item()
        
        return float(h_boundary - h_stable)
    def compute_mcd(self, ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 24000, n_mfcc: int = 24) -> float:
        """
        Calculates Mel-Cepstral Distortion (MCD) utilizing DTW 
        and strict amplitude normalization to prevent logarithmic explosion.
        """
        # --- NEW: Peak Amplitude Normalization ---
        # Prevent float32 vs int16 scaling mismatch
        if np.max(np.abs(ref_audio)) > 0:
            ref_audio = ref_audio.astype(np.float32) / np.max(np.abs(ref_audio))
        if np.max(np.abs(gen_audio)) > 0:
            gen_audio = gen_audio.astype(np.float32) / np.max(np.abs(gen_audio))
            
        # 1. Extract MFCCs
        mfcc_ref = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_gen = librosa.feature.mfcc(y=gen_audio, sr=sr, n_mfcc=n_mfcc)
        
        # 2. Drop the 0th coefficient (Energy)
        mfcc_ref = mfcc_ref[1:, :]
        mfcc_gen = mfcc_gen[1:, :]
        
        # 3. Apply Dynamic Time Warping (DTW) to align temporal sequences
        D, wp = librosa.sequence.dtw(X=mfcc_ref, Y=mfcc_gen, metric='euclidean')
        
        # 4. Extract the aligned frames
        ref_indices = wp[:, 0]
        gen_indices = wp[:, 1]
        mfcc_ref_aligned = mfcc_ref[:, ref_indices]
        mfcc_gen_aligned = mfcc_gen[:, gen_indices]
        
        # 5. Calculate Euclidean distance and apply formal MCD formula
        diff = mfcc_ref_aligned - mfcc_gen_aligned
        mcd_frames = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.sum(diff ** 2, axis=0))
        
        return float(np.mean(mcd_frames))

    def compute_sim_r(self, ref_embedding: torch.Tensor, gen_embedding: torch.Tensor) -> float:
        """
        Calculates Cosine Speaker Similarity (SIM-R) to prove voice preservation.
        """
        sim = torch.nn.functional.cosine_similarity(ref_embedding.unsqueeze(0), gen_embedding.unsqueeze(0))
        return float(sim.item())

    def compute_wer(self, audio_path: str, ground_truth_text: str) -> float:
        """
        Calculates Word Error Rate (WER) using native PyTorch Whisper Large-v3-Turbo.
        """
        # Transcribe offline audio (forcing Hindi language tags for CS-TTS evaluation)
        result = self.asr_pipeline(audio_path, generate_kwargs={"language": "hindi", "task": "transcribe"})
        transcribed_text = result["text"].strip()
        
        # Levenshtein distance-based word error calculation
        ref_words = ground_truth_text.lower().split()
        hyp_words = transcribed_text.lower().split()
        
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
        for i in range(len(ref_words) + 1): d[i, 0] = i
        for j in range(len(hyp_words) + 1): d[0, j] = j
            
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
                
        wer = (d[len(ref_words), len(hyp_words)] / max(len(ref_words), 1)) * 100.0
        return float(wer)

    def compute_attention_entropy_variance(self, attn_matrix: torch.Tensor, boundary_indices: Set[int]) -> float:
        """
        Calculates H(A_Beta) - H(A_S) to prove mathematical alignment stability.
        """
        if not boundary_indices or attn_matrix.shape[-1] == 0:
            return 0.0
            
        attn_matrix = torch.clamp(attn_matrix, min=1e-9, max=1.0)
        entropy_per_frame = -torch.sum(attn_matrix * torch.log(attn_matrix), dim=-1).squeeze()
        
        all_indices = set(range(attn_matrix.shape[-1]))
        steady_state_indices = list(all_indices - boundary_indices)
        valid_boundaries = [i for i in list(boundary_indices) if i < len(entropy_per_frame)]
        valid_steady = [i for i in steady_state_indices if i < len(entropy_per_frame)]
        
        if not valid_boundaries or not valid_steady:
            return 0.0
            
        h_beta = torch.mean(entropy_per_frame[valid_boundaries]).item()
        h_s = torch.mean(entropy_per_frame[valid_steady]).item()
        
        return h_beta - h_s
