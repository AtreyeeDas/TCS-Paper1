import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.stage1_phonetics import PhoneticUnificationEngine
from src.stage2_speaker import SpeakerConditioningEncoder

class HinglishCodeSwitchedDataset(Dataset):
    def __init__(self, metadata_path: str, audio_dir: str):
        self.audio_dir = audio_dir
        self.data = []
        
        # Load your cleaned transcripts
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    self.data.append({"utt_id": parts[0], "text": parts[1]})
                    
        self.phonetic_engine = PhoneticUnificationEngine()
        self.speaker_encoder = SpeakerConditioningEncoder()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = os.path.join(self.audio_dir, f"{item['utt_id']}.wav")
        
        # 1. Process Text & Get Boundaries (Stage 1 of your paper)
        ipa_tokens, boundaries = self.phonetic_engine.process_text(item["text"])
        
        # 2. Extract Speaker Embedding (Stage 2 of your paper)
        speaker_emb = self.speaker_encoder.extract_embedding(audio_path)
        
        # 3. Load Target Audio for TTS Loss
        wav, _ = librosa.load(audio_path, sr=24000)
        
        return {
            "ipa_tokens": ipa_tokens,
            "boundaries": torch.tensor(list(boundaries), dtype=torch.long),
            "speaker_embedding": speaker_emb,
            "target_audio": torch.tensor(wav, dtype=torch.float32)
        }
