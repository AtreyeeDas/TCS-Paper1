import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HinglishCodeSwitchedDataset
from src.stage3_4_engine import PhonemeAlignedEntropyLoss, compute_total_loss
from src.model_architecture import CodeSwitchedTTSModel

def train_entropy_regularized_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Initializing Training on {device} (Blackwell / RTX PRO 5000 Optimized)")
    
    # 1. Initialize Datasets & Loaders
    train_dataset = HinglishCodeSwitchedDataset(
        metadata_path="./MUCS_sliced/train/transcripts.txt", 
        audio_dir="./MUCS_sliced/train/audio"
    )
    val_dataset = HinglishCodeSwitchedDataset(
        metadata_path="./MUCS_sliced/test/transcripts.txt", 
        audio_dir="./MUCS_sliced/test/audio"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 2. Initialize Model & Objectives
    model = CodeSwitchedTTSModel().to(device)
    entropy_criterion = PhonemeAlignedEntropyLoss(eps=1e-9)
    tts_criterion = torch.nn.L1Loss()
    
    # 3. Freeze Base Weights; Unfreeze ONLY Cross-Attention
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "cross_attention" in name or "audio_head" in name:
            param.requires_grad = True
            
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    print("[+] Starting Fine-Tuning & Validation Loop...")
    
    for epoch in range(10):
        # --- TRAINING ---
        model.train()
        train_loss, train_entropy = 0.0, 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            ipa_tokens = batch["ipa_tokens"].to(device)
            boundaries_list = [set(b.numpy()) for b in batch["boundaries"]]
            speaker_emb = batch["speaker_embedding"].to(device)
            target_audio = batch["target_audio"].to(device)
            
            # Pad target audio to match model output dimensions (24000 samples)
            if target_audio.shape[1] < 24000:
                target_audio = torch.nn.functional.pad(target_audio, (0, 24000 - target_audio.shape[1]))
            else:
                target_audio = target_audio[:, :24000]
                
            predicted_audio, cross_attn_matrix = model(ipa_tokens, speaker_emb)
            
            loss_tts = tts_criterion(predicted_audio, target_audio)
            
            # Compute average entropy loss across batch items
            loss_entropy = torch.mean(torch.stack([
                entropy_criterion(cross_attn_matrix[i:i+1], boundaries_list[i])
                for i in range(len(boundaries_list))
            ]))
            
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
                
                if target_audio.shape[1] < 24000:
                    target_audio = torch.nn.functional.pad(target_audio, (0, 24000 - target_audio.shape[1]))
                else:
                    target_audio = target_audio[:, :24000]
                    
                predicted_audio, cross_attn_matrix = model(ipa_tokens, speaker_emb)
                
                loss_tts = tts_criterion(predicted_audio, target_audio)
                loss_entropy = torch.mean(torch.stack([
                    entropy_criterion(cross_attn_matrix[i:i+1], boundaries_list[i])
                    for i in range(len(boundaries_list))
                ]))
                
                loss_total = compute_total_loss(loss_tts, loss_entropy, lamda=0.1)
                val_loss += loss_total.item()
                val_entropy += loss_entropy.item()
                
        # --- LOGGING ---
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} (Ent: {train_entropy/len(train_loader):.4f}) | Val Loss: {val_loss/len(val_loader):.4f} (Ent: {val_entropy/len(val_loader):.4f})")
        
    print("[✓] Training Complete. Saving checkpoint for Ablation testing...")
    torch.save(model.state_dict(), "icassp_model_final.pth")

if __name__ == "__main__":
    train_entropy_regularized_model()
