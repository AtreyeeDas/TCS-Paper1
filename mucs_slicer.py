import os
import librosa
import soundfile as sf
from tqdm import tqdm

def process_split(split_name: str, base_dataset_dir: str, output_base_dir: str):
    """
    Slices Kaldi-formatted audio into sentence-level utterances for a given dataset split.
    """
    kaldi_dir = os.path.join(base_dataset_dir, split_name, "transcripts")
    audio_dir = os.path.join(base_dataset_dir, split_name, "audio")
    output_audio_dir = os.path.join(output_base_dir, split_name, "audio")
    output_meta_path = os.path.join(output_base_dir, split_name, "transcripts.txt")
    
    os.makedirs(output_audio_dir, exist_ok=True)

    # 1. Parse wav.scp
    wav_scp_path = os.path.join(kaldi_dir, "wav.scp")
    recording_to_path = {}
    with open(wav_scp_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                recording_to_path[parts[0]] = parts[1]

    # 2. Parse text transcripts
    text_path = os.path.join(kaldi_dir, "text")
    utt_to_text = {}
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt_to_text[parts[0]] = parts[1]

    # 3. Parse segments
    segments_path = os.path.join(kaldi_dir, "segments")
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

    print(f"[+] [{split_name.upper()}] Found {len(segments)} utterances. Slicing audio...")
    
    rec_to_segments = {}
    for seg in segments:
        rec_to_segments.setdefault(seg["rec_id"], []).append(seg)

    valid_utterances = []

    for rec_id, segs in tqdm(rec_to_segments.items(), desc=f"Processing {split_name}"):
        if rec_id not in recording_to_path:
            continue
            
        audio_filename = os.path.basename(recording_to_path[rec_id])
        if not audio_filename.endswith('.wav'):
            audio_filename += '.wav'
            
        audio_path = os.path.join(audio_dir, audio_filename)
        if not os.path.exists(audio_path):
            continue
            
        try:
            y, sr = librosa.load(audio_path, sr=24000)
            for seg in segs:
                start_sample = int(seg["start"] * sr)
                end_sample = int(seg["end"] * sr)
                sliced_audio = y[start_sample:end_sample]
                
                out_path = os.path.join(output_audio_dir, f"{seg['utt_id']}.wav")
                sf.write(out_path, sliced_audio, sr)
                
                if seg['utt_id'] in utt_to_text:
                    valid_utterances.append(f"{seg['utt_id']}\t{utt_to_text[seg['utt_id']]}\n")
        except Exception as e:
            print(f"[-] Failed on recording {rec_id}: {e}")

    with open(output_meta_path, "w", encoding="utf-8") as f:
        f.writelines(valid_utterances)

    print(f"[✓] [{split_name.upper()}] Saved {len(valid_utterances)} sliced files to {output_audio_dir}")

if __name__ == "__main__":
    BASE_DATASET = "/home/spark2/Models/MUCS_Hindi-English_dataset"
    OUTPUT_DIR = "./MUCS_sliced"
    
    process_split("train", BASE_DATASET, OUTPUT_DIR)
    process_split("test", BASE_DATASET, OUTPUT_DIR)
