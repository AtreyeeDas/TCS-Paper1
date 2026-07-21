import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionExtractorLayer(nn.Module):
    """
    Custom Transformer Decoder layer configured to explicitly intercept and return
    the Cross-Attention Probability Matrix A for entropy regularization.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, speaker_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Inject continuous speaker conditioning bias into text memory space
        memory_conditioned = memory + speaker_bias.unsqueeze(1)
        
        # 1. Autoregressive Self-Attention
        attn_out, _ = self.self_attn(x, x, x, is_causal=True)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 2. Phoneme-Aligned Cross-Attention (Intercept Attention Matrix A)
        cross_out, attn_weights = self.cross_attn(
            query=x, 
            key=memory_conditioned, 
            value=memory_conditioned, 
            need_weights=True, 
            average_attn_weights=True  # Returns matrix A of shape [Batch, M, T]
        )
        x = self.norm2(x + self.dropout(cross_out))
        
        # 3. Feed Forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x, attn_weights


class PhonemeAlignedEntropyLoss(nn.Module):
    """
    Stage 4: Mathematical Shannon Entropy loss evaluated strictly at linguistic
    code-switching boundaries (Beta) to prevent cross-attention scattering.
    """
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, attn_matrix: torch.Tensor, boundary_indices: set[int]) -> torch.Tensor:
        """
        Args:
            attn_matrix: Extracted probability tensor A of shape [Batch, M_frames, T_phonemes].
            boundary_indices: Set of integer positions Beta where script transitions occur.
        Returns:
            l_entropy: Scalar tensor representing the boundary Shannon entropy penalty.
        """
        if not boundary_indices or len(boundary_indices) == 0:
            # Return zero gradient-enabling loss if no intra-sentential boundary exists
            return torch.tensor(0.0, device=attn_matrix.device, requires_grad=True)

        # Ensure valid probability distribution over phoneme sequence T
        attn_matrix = torch.clamp(attn_matrix, min=self.eps, max=1.0)
        
        # Calculate Shannon Entropy per frame m: H(A_m) = - Sum_t (A_{m,t} * log(A_{m,t}))
        entropy_per_frame = -torch.sum(attn_matrix * torch.log(attn_matrix), dim=-1) # Shape: [Batch, M]
        
        # Filter strictly for target boundary frames in Beta
        boundary_list = list(boundary_indices)
        
        # Guard against index out-of-bounds if phoneme sequence length T was truncated
        valid_boundaries = [idx for idx in boundary_list if idx < attn_matrix.shape[-1]]
        
        if not valid_boundaries:
            return torch.tensor(0.0, device=attn_matrix.device, requires_grad=True)
            
        # Select boundary transition frames across all batches
        # We average entropy over frames corresponding to transition indices
        boundary_entropy = entropy_per_frame[:, valid_boundaries]
        l_entropy = torch.mean(boundary_entropy)
        
        return l_entropy

# Optimize execution on modern architectures using Torch Autotune as specified in execution plan
@torch.compile(mode="max-autotune")
def compute_total_loss(l_tts: torch.Tensor, l_entropy: torch.Tensor, lamda: float = 0.5) -> torch.Tensor:
    """Computes total training objective: L_total = L_tts + lamda * L_entropy"""
    return l_tts + (lamda * l_entropy)
