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
