import os
import torch
import torch.optim as optim
from dataset import HinglishCodeSwitchedDataset
from src.model_architecture import CodeSwitchedTTSModel
from src.stage3_4_engine import PhonemeAlignedEntropyLoss, compute_total_loss
from torch.utils.data import DataLoader


# --- NEW: CUSTOM COLLATE FUNCTION TO HANDLE VARIABLE-LENGTH AUDIO & TEXT ---
def custom_tts_collate_fn(batch):
    """Pads variable-length IPA text sequences and audio waveforms into uniform batch tensors."""
    # 1. Pad IPA tokens to the maximum length found in this specific batch
    max_ipa_len = max([item["ipa_tokens"].shape[0] for item in batch])
    padded_ipa = torch.zeros(len(batch), max_ipa_len, dtype=torch.long)
    for i, item in enumerate(batch):
        padded_ipa[i, : item["ipa_tokens"].shape[0]] = item["ipa_tokens"]

    # 2. Pad or truncate Target Audio to exactly 24000 samples (1 second of 24kHz audio)
    padded_audio = torch.zeros(len(batch), 24000, dtype=torch.float32)
    for i, item in enumerate(batch):
        wav = item["target_audio"]
        if wav.shape[0] > 24000:
            padded_audio[i] = wav[:24000]
        else:
            padded_audio[i, : wav.shape[0]] = wav

    # 3. Stack speaker embeddings uniformly -> Shape: [Batch, 192]
    spk_embs = torch.stack(
        [
            item["speaker_embedding"].squeeze(0)
            if item["speaker_embedding"].dim() > 1
            else item["speaker_embedding"]
            for item in batch
        ],
        dim=0,
    )

    # 4. Keep boundaries as a raw Python list of tensors to prevent stacking crashes
    boundaries_list = [item["boundaries"] for item in batch]

    return {
        "ipa_tokens": padded_ipa,
        "target_audio": padded_audio,
        "speaker_embedding": spk_embs,
        "boundaries": boundaries_list,
    }


# -----------------------------------------------------------------------------


def train_entropy_regularized_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Initializing Training on {device} (Blackwell / RTX PRO 5000 Optimized)")

    # 1. Initialize Datasets & Loaders WITH CUSTOM COLLATE FUNCTION
    train_dataset = HinglishCodeSwitchedDataset(
        metadata_path="./MUCS_sliced/train/transcripts.txt",
        audio_dir="./MUCS_sliced/train/audio",
    )
    val_dataset = HinglishCodeSwitchedDataset(
        metadata_path="./MUCS_sliced/test/transcripts.txt",
        audio_dir="./MUCS_sliced/test/audio",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=custom_tts_collate_fn,  # <-- Injected here!
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_tts_collate_fn,  # <-- Injected here!
    )

    # 2. Initialize Model & Objectives
    model = CodeSwitchedTTSModel().to(device)
    entropy_criterion = PhonemeAlignedEntropyLoss(eps=1e-9)
    tts_criterion = torch.nn.L1Loss()

    # 3. Freeze Base Weights; Unfreeze ONLY Cross-Attention & Head
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "cross_attention" in name or "audio_head" in name:
            param.requires_grad = True

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    print("[+] Starting Fine-Tuning & Validation Loop...")

    for epoch in range(10):
        # --- TRAINING ---
        model.train()
        train_loss, train_entropy = 0.0, 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            ipa_tokens = batch["ipa_tokens"].to(device)
            # Convert boundary tensors back into Python sets for Stage 4 entropy evaluation
            boundaries_list = [set(b.numpy()) for b in batch["boundaries"]]
            speaker_emb = batch["speaker_embedding"].to(device)
            target_audio = batch["target_audio"].to(device)

            predicted_audio, cross_attn_matrix = model(ipa_tokens, speaker_emb)

            loss_tts = tts_criterion(predicted_audio, target_audio)

            # Compute average entropy loss across batch items
            loss_entropy = torch.mean(
                torch.stack([
                    entropy_criterion(
                        cross_attn_matrix[i : i + 1], boundaries_list[i]
                    )
                    for i in range(len(boundaries_list))
                ])
            )

            loss_total = compute_total_loss(loss_tts, loss_entropy, lamda=0.1)
            loss_total.backward()
            optimizer.step()

            train_loss += loss_total.item()
            train_entropy += loss_entropy.item()

        # --- VALIDATION ---
        model.eval()
        val_loss, val_entropy = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                ipa_tokens = batch["ipa_tokens"].to(device)
                boundaries_list = [set(b.numpy()) for b in batch["boundaries"]]
                speaker_emb = batch["speaker_embedding"].to(device)
                target_audio = batch["target_audio"].to(device)

                predicted_audio, cross_attn_matrix = model(
                    ipa_tokens, speaker_emb
                )

                loss_tts = tts_criterion(predicted_audio, target_audio)
                loss_entropy = torch.mean(
                    torch.stack([
                        entropy_criterion(
                            cross_attn_matrix[i : i + 1], boundaries_list[i]
                        )
                        for i in range(len(boundaries_list))
                    ])
                )

                loss_total = compute_total_loss(
                    loss_tts, loss_entropy, lamda=0.1
                )
                val_loss += loss_total.item()
                val_entropy += loss_entropy.item()

        # --- LOGGING ---
        print(
            f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} (Ent: {train_entropy/len(train_loader):.4f}) | Val Loss: {val_loss/len(val_loader):.4f} (Ent: {val_entropy/len(val_loader):.4f})"
        )

    print("[✓] Training Complete. Saving checkpoint for Ablation testing...")
    torch.save(model.state_dict(), "icassp_model_final.pth")


if __name__ == "__main__":
    train_entropy_regularized_model()
