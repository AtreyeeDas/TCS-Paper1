import numpy as np
import torch
import torch.nn as nn

class EnergyGatedAcousticGuardrail:
    """
    Stage 5: Deterministic signal processing filter computing Short-Time Energy (STE).
    Updated with clinical thresholds to preserve intelligibility (WER).
    """
    # Increased window to 50ms and dropped threshold to 1e-6 (near silence)
    def __init__(self, sample_rate: int = 24000, window_ms: int = 50, energy_threshold: float = 1e-6):
        self.sr = sample_rate
        self.window_size = int((sample_rate * window_ms) / 1000)
        self.threshold = energy_threshold

    def apply_guardrail(self, waveform: np.ndarray, text_completion_idx: int = None) -> np.ndarray:
        if len(waveform) < self.window_size:
            return waveform

        # Scan from 85% of audio to ensure we don't clip the main speech body
        start_scan = text_completion_idx if text_completion_idx is not None else int(len(waveform) * 0.85)
        
        num_windows = (len(waveform) - start_scan) // self.window_size
        if num_windows <= 0:
            return waveform

        # Normalize waveform internally before calculating energy to ensure scale-invariance
        norm_wave = waveform.astype(np.float32)
        if np.max(np.abs(norm_wave)) > 0:
            norm_wave = norm_wave / np.max(np.abs(norm_wave))

        for i in range(num_windows):
            idx_start = start_scan + (i * self.window_size)
            idx_end = idx_start + self.window_size
            window = norm_wave[idx_start:idx_end]
            
            # Compute STE on normalized array
            ste = np.mean(window ** 2)
            
            if ste < self.threshold:
                return waveform[:idx_start]
                
        return waveform