import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends= lambda: ["soundfile"]
import torch
import torch.nn as nn
from speechbrain.inference.speaker import EncoderClassifier
class SpeakerConditioningEncoder(nn.Module):
    """
    Stage 2: Extracts a fixed-dimensional speaker embedding vector from a short
    3-5 second reference audio prompt for zero-shot speaker adaptation.
    """
    def __init__(self, model_source: str = "/home/spark2/Models/spkrec-ecapa-voxceleb", device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Initialize frozen ECAPA-TDNN embedding extractor
        self.classifier = EncoderClassifier.from_hparams(
            source=model_source, 
            run_opts={"device": str(self.device), "local_files_only": True}
        )
        self.classifier.eval()

    @torch.no_grad()
    def extract_embedding(self, audio_path: str) -> torch.Tensor:
        """
        Loads reference audio, verifies sampling constraints, and returns a normalized
        speaker embedding vector injected into downstream cross-attention layers.
        """
        signal, fs = torchaudio.load(audio_path)
        
        # Resample to 16kHz if required by ECAPA-TDNN backbone
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)
            
        # Convert stereo to mono if applicable
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
            
        # Ensure duration is within clinical/practical zero-shot limits (3-5s)
        max_len = 16000 * 6  # 6 seconds maximum buffer
        if signal.shape[1] > max_len:
            signal = signal[:, :max_len]
            
        signal = signal.to(self.device)
        embedding = self.classifier.encode_batch(signal)
        # Normalize continuous embedding vector
        embedding = torch.nn.functional.normalize(embedding.squeeze(), p=2, dim=-0)
        return embedding
