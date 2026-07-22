import numpy as np
import librosa

class EnergyGatedAcousticGuardrail:
    """
    Stage 5: Post-vocoder acoustic guardrail that prevents artifact explosions
    and energy discontinuities at intra-sentential language switching boundaries.
    """
    def __init__(self, sr: int = 24000, frame_length: int = 1024, hop_length: int = 256):
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.max_db_threshold = 3.0 # Maximum allowable dB jump between adjacent frames

    def apply_guardrail(self, waveform: np.ndarray) -> np.ndarray:
        """
        Applies RMS energy gating and envelope smoothing across the audio waveform.
        """
        if len(waveform) == 0:
            return waveform

        # 1. Compute RMS energy envelope
        rms = librosa.feature.rms(y=waveform, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        db_envelope = librosa.amplitude_to_db(rms, ref=np.max)
        
        # 2. Identify sudden energy discontinuities (Spikes > threshold)
        db_diff = np.diff(db_envelope, prepend=db_envelope[0])
        spike_indices = np.where(db_diff > self.max_db_threshold)[0]
        
        if len(spike_indices) == 0:
            return waveform
            
        # 3. Apply Localized Gain Attenuation at artifact locations
        modified_waveform = waveform.copy()
        for idx in spike_indices:
            sample_start = max(0, int(idx * self.hop_length - self.frame_length // 2))
            sample_end = min(len(waveform), int(idx * self.hop_length + self.frame_length // 2))
            
            # Smooth transition using a Hanning fade window
            window = np.hanning(sample_end - sample_start)
            attenuation_factor = 10 ** (-self.max_db_threshold / 20.0)
            modified_waveform[sample_start:sample_end] *= (1.0 - (1.0 - attenuation_factor) * window)
            
        return np.clip(modified_waveform, -1.0, 1.0)
