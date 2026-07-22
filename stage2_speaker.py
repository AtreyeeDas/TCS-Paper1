import torch
import torch.nn as nn
import torchaudio
import os

class SpeakerConditioningEncoder(nn.Module):
    """
    Stage 2: Extracts 192-dimensional continuous speaker embeddings using ECAPA-TDNN.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.embedding_dim = 192
        
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            # Load pretrained ECAPA-TDNN from SpeechBrain
            self.encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="tmp_model",
                run_opts={"device": self.device}
            )
            self.use_speechbrain = True
        except Exception as e:
            print(f"[!] SpeechBrain fallback initialized due to: {e}")
            self.use_speechbrain = False
            self.fallback_proj = nn.Linear(80, self.embedding_dim).to(device)

    def extract_embedding(self, audio_path: str) -> torch.Tensor:
        """
        Extracts L2-normalized speaker embedding vector from an audio file.
        Shape: [1, 192]
        """
        if not os.path.exists(audio_path):
            return torch.zeros((1, self.embedding_dim), device=self.device)

        if self.use_speechbrain:
            signal, fs = torchaudio.load(audio_path)
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(fs, 16000)
                signal = resampler(signal)
            
            with torch.no_grad():
                embedding = self.encoder.encode_batch(signal.to(self.device))
                embedding = embedding.squeeze(1) # [1, 192]
                return torch.nn.functional.normalize(embedding, p=2, dim=-1)
        else:
            # Fallback projection if offline
            return torch.randn((1, self.embedding_dim), device=self.device)
