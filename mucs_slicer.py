import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm

def slice_mucs_audio(long_wav_path: str, transcript_tsv_path: str, output_dir: str):
    """
    Slices a long MUCS conversational .wav file into sentence-level clips 
    based on start and end timestamps.
    """
    print(f"[+] Loading massive audio file: {long_wav_path}...")
    # Load with librosa to ensure 24kHz target sample rate
    y, sr = librosa.load(long_wav_path, sr=24000)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Assuming transcript is tab-separated: Utt_ID \t Start \t End \t Text
    df = pd.read_csv(transcript_tsv_path, sep='\t', names=['utt_id', 'start', 'end', 'text'])
    
    print(f"[+] Slicing {len(df)} sentences...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            start_sample = int(float(row['start']) * sr)
            end_sample = int(float(row['end']) * sr)
            
            # Slice array
            sliced_audio = y[start_sample:end_sample]
            
            # Save individual sentence
            out_path = os.path.join(output_dir, f"{row['utt_id']}.wav")
            sf.write(out_path, sliced_audio, sr)
            
        except Exception as e:
            print(f"[-] Failed to slice {row['utt_id']}: {e}")
            
    print(f"[✓] Slicing complete. Files saved to {output_dir}")

if __name__ == "__main__":
    # UPDATE THESE PATHS based on your MUCS folder structure
    slice_mucs_audio(
        long_wav_path="./MUCS/audio/long_conversation_1.wav", 
        transcript_tsv_path="./MUCS/transcripts/text_with_timestamps.tsv",
        output_dir="./MUCS_sliced/audio"
    )
