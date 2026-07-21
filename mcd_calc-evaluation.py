    def compute_mcd(self, ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 24000, n_mfcc: int = 24) -> float:
        """
        Calculates Mel-Cepstral Distortion (MCD) in decibels [dB] utilizing DTW 
        to ensure proper temporal alignment before Euclidean distance calculation.
        """
        # 1. Extract MFCCs
        mfcc_ref = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_gen = librosa.feature.mfcc(y=gen_audio, sr=sr, n_mfcc=n_mfcc)
        
        # 2. Drop the 0th coefficient (Energy) as per standard MCD literature
        mfcc_ref = mfcc_ref[1:, :]
        mfcc_gen = mfcc_gen[1:, :]
        
        # 3. Apply Dynamic Time Warping (DTW) to align the two sequences temporally
        D, wp = librosa.sequence.dtw(X=mfcc_ref, Y=mfcc_gen, metric='euclidean')
        
        # wp is a 2D array containing the alignment path indices
        ref_indices = wp[:, 0]
        gen_indices = wp[:, 1]
        
        # 4. Extract the aligned frames
        mfcc_ref_aligned = mfcc_ref[:, ref_indices]
        mfcc_gen_aligned = mfcc_gen[:, gen_indices]
        
        # 5. Calculate the Euclidean distance across the aligned frames
        diff = mfcc_ref_aligned - mfcc_gen_aligned
        
        # 6. Apply the formal MCD formula over the DTW path
        # MCD = (10 / ln(10)) * sqrt(2 * sum( (c_t - c_hat_t)^2 ))
        mcd_frames = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.sum(diff ** 2, axis=0))
        
        return float(np.mean(mcd_frames))
