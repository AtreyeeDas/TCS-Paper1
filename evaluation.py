import librosa
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Set

class EvaluationSuite:
    """
    Applies objective validation metrics across all 4 Ablation Arms using
    100% native PyTorch components compatible with CUDA 13.0 Blackwell architecture.
    """
    def __init__(self, whisper_model_path: str = "/home/username/cs_tts_pipeline/assets/whisper_turbo", device: str = "cuda"):
        self.device = device
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Load native PyTorch Whisper Model offline
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper_model_path, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True,
            local_files_only=True
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(whisper_model_path, local_files_only=True)
        
        # Initialize native Hugging Face ASR pipeline with PyTorch SDPA acceleration
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=device
        )

    def compute_mcd(self, ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 24000, n_mfcc: int = 24) -> float:
        """
        Calculates Mel-Cepstral Distortion (MCD) in decibels [dB] between generated
        and ground-truth speech to verify absence of acoustic babble.
        """
        mfcc_ref = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_gen = librosa.feature.mfcc(y=gen_audio, sr=sr, n_mfcc=n_mfcc)
        
        min_frames = min(mfcc_ref.shape[1], mfcc_gen.shape[1])
        diff = mfcc_ref[:, :min_frames] - mfcc_gen[:, :min_frames]
        
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
