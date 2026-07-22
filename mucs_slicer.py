import os
import librosa
import soundfile as sf
from tqdm import tqdm

def process_kaldi_directory(mucs_kaldi_dir: str, output_audio_dir: str):
    """
    Parses standard Kaldi formatted files (wav.scp, segments) to slice 
    massive conversational audio into sentence-level ground truths.
    """
    os.makedirs(output_audio_dir, exist_ok=True)

    # 1. Parse wav.scp (Maps Recording ID -> File Path)
    wav_scp_path = os.path.join(mucs_kaldi_dir, "wav.scp")
    recording_to_path = {}
    with open(wav_scp_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                recording_to_path[parts[0]] = parts[1]

    # 2. Parse segments (Maps Utterance ID -> Recording ID, Start, End)
    segments_path = os.path.join(mucs_kaldi_dir, "segments")
    segments = []
    with open(segments_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                segments.append({
                    "utt_id": parts[0],
                    "rec_id": parts[1],
                    "start": float(parts[2]),
                    "end": float(parts[3])
                })

    print(f"[+] Found {len(segments)} utterances to slice. Loading audio files...")

    # Group segments by recording to avoid reloading massive audio files multiple times
    rec_to_segments = {}
    for seg in segments:
        rec_to_segments.setdefault(seg["rec_id"], []).append(seg)

    for rec_id, segs in tqdm(rec_to_segments.items(), desc="Processing Recordings"):
        if rec_id not in recording_to_path:
            continue
            
        audio_path = recording_to_path[rec_id]
        
        # --- PATH FIX: Point to your new audio/ subfolder ---
        audio_filename = os.path.basename(audio_path)
        if not audio_filename.endswith('.wav'):
            audio_filename += '.wav'
            
        # Hardcoding the exact path to your new audio folder
        actual_audio_folder = "/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/audio"
        audio_path = os.path.join(actual_audio_folder, audio_filename)
        # ----------------------------------------------------
            
        try:
            # Load the full audio file into RAM at 24kHz
            y, sr = librosa.load(audio_path, sr=24000)
            
            for seg in segs:
                start_sample = int(seg["start"] * sr)
                end_sample = int(seg["end"] * sr)
                
                sliced_audio = y[start_sample:end_sample]
                out_path = os.path.join(output_audio_dir, f"{seg['utt_id']}.wav")
                
                sf.write(out_path, sliced_audio, sr)
                
        except Exception as e:
            print(f"[-] Failed to process recording {rec_id}: {e}")

    print(f"[✓] Slicing complete! All files saved to {os.path.abspath(output_audio_dir)}")

if __name__ == "__main__":
    # Point this to the subfolder containing text, segments, wav.scp, etc.
    KALDI_DIR = "/home/spark2/Models/MUCS_Hindi-English_test_dataset/test/transcripts" 
    
    # This is where all the thousands of tiny sliced .wav files will be saved
    OUTPUT_DIR = "./MUCS_sliced/audio"
    
    process_kaldi_directory(KALDI_DIR, OUTPUT_DIR)
