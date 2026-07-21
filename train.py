import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HinglishCodeSwitchedDataset
from src.stage3_4_engine import PhonemeAlignedEntropyLoss

# NOTE: You will need to import your actual model architecture here.
# If you are fine-tuning XTTS, you wrap it. If you built a custom transformer, instantiate it.
from your_model_architecture import CodeSwitchedTTSModel 

def train_entropy_regularized_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Initializing Training on {device} (Blackwell / RTX PRO 5000)")
    
    # 1. Initialize Dataset & Loaders
    dataset = HinglishCodeSwitchedDataset(
        metadata_path="./MUCS_sliced/transcripts.txt", 
        audio_dir="./MUCS_sliced/audio"
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. Initialize Model & Your Custom Loss
    model = CodeSwitchedTTSModel().to(device)
    entropy_criterion = PhonemeAlignedEntropyLoss(lambda_weight=0.1)
    tts_criterion = torch.nn.L1Loss() # Standard Mel-Reconstruction Loss
    
    # 3. Freeze Base Weights, Unfreeze ONLY Cross-Attention (Crucial for VRAM & stability)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "cross_attention" in name:
            param.requires_grad = True
            
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    print("[+] Starting Fine-Tuning Loop...")
    model.train()
    
    for epoch in range(10): # Adjust epochs as needed
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Move data to GPU
            ipa_tokens = batch["ipa_tokens"].to(device)
            boundaries = batch["boundaries"].to(device)
            speaker_emb = batch["speaker_embedding"].to(device)
            target_audio = batch["target_audio"].to(device)
            
            # Forward Pass: Your model MUST return the attention matrix alongside the audio
            predicted_audio, cross_attn_matrix = model(
                text_tokens=ipa_tokens, 
                speaker_embedding=speaker_emb
            )
            
            # Calculate Standard TTS Loss
            loss_tts = tts_criterion(predicted_audio, target_audio)
            
            # Calculate YOUR Novel Entropy Loss
            loss_entropy = entropy_criterion(cross_attn_matrix, boundaries)
            
            # Combined Objective Function
            loss_total = loss_tts + loss_entropy
            
            # Backpropagation
            loss_total.backward()
            optimizer.step()
            
            total_loss += loss_total.item()
            
        print(f"Epoch {epoch+1} | Total Loss: {total_loss/len(dataloader):.4f} | Entropy Loss: {loss_entropy.item():.4f}")
        
    print("[✓] Fine-Tuning Complete. Saving checkpoint...")
    torch.save(model.state_dict(), "icassp_model_final.pth")

if __name__ == "__main__":
    train_entropy_regularized_model()
