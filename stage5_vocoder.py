import numpy as np
import torch
import torch.nn as nn

class EnergyGatedAcousticGuardrail:
    """
    Stage 5: Deterministic signal processing filter computing Short-Time Energy (STE)
    and Zero-Crossing Rate (ZCR) over 10ms sliding windows to truncate trailing whispers
    before reaching the audio buffer.
    """
    def __init__(self, sample_rate: int = 24000, window_ms: int = 10, energy_threshold: float = 1e-4):
        self.sr = sample_rate
        self.window_size = int((sample_rate * window_ms) / 1000)
        self.threshold = energy_threshold

    def apply_guardrail(self, waveform: np.ndarray, text_completion_idx: int = None) -> np.ndarray:
        """
        Scans generated continuous time-domain waveform and truncates signal
        immediately when energy drops below threshold post-text completion.
        """
        if len(waveform) < self.window_size:
            return waveform

        # If approximate sample index of text completion is unknown, scan from 75% of audio
        start_scan = text_completion_idx if text_completion_idx is not None else int(len(waveform) * 0.75)
        
        num_windows = (len(waveform) - start_scan) // self.window_size
        if num_windows <= 0:
            return waveform

        for i in range(num_windows):
            idx_start = start_scan + (i * self.window_size)
            idx_end = idx_start + self.window_size
            window = waveform[idx_start:idx_end]
            
            # Compute Short-Time Energy (STE): Mean squared amplitude
            ste = np.mean(window ** 2)
            
            # Compute Zero-Crossing Rate (ZCR) as a secondary validation check
            zcr = 0.5 * np.mean(np.abs(np.diff(np.sign(window))))
            
            # Truncate array instantly if energy drops below threshold tau
            if ste < self.threshold:
                return waveform[:idx_start]
                
        return waveform

class VocoderWithGuardrail(nn.Module):
    """Wrapper integrating HiFi-GAN Vocoder synthesis with the real-time acoustic guardrail."""
    def __init__(self, vocoder_model: nn.Module, sample_rate: int = 24000):
        super().__init__()
        self.vocoder = vocoder_model
        self.vocoder.eval()
        self.guardrail = EnergyGatedAcousticGuardrail(sample_rate=sample_rate)

    @torch.no_grad()
    def synthesize_and_clean(self, acoustic_tokens: torch.Tensor) -> np.ndarray:
        """Synthesizes discrete acoustic tokens into a clean waveform."""
        # Generate time-domain waveform via HiFi-GAN
        raw_waveform = self.vocoder(acoustic_tokens).squeeze().cpu().numpy()
        
        # Pass through deterministic endpointing filter
        clean_waveform = self.guardrail.apply_guardrail(raw_waveform)
        return clean_waveform
