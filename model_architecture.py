import torch
import torch.nn as nn
from src.stage3_4_engine import CrossAttentionExtractorLayer

class CodeSwitchedTTSModel(nn.Module):
    """
    Acoustic Transformer model configured to expose cross-attention matrices
    for Stage 4 Phoneme-Aligned Entropy Regularization.
    """
    def __init__(self, vocab_size: int = 256, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048):
        super().__init__()
        self.d_model = d_model
        
        # Phoneme Embedding & Speaker Projection
        self.phoneme_embedding = nn.Embedding(vocab_size, d_model)
        self.speaker_proj = nn.Linear(192, d_model)
        
        # Decoder with Cross-Attention Interception
        self.cross_attention_layer = CrossAttentionExtractorLayer(d_model, n_heads, d_ff)
        
        # Acoustic Output Projection (Predicting raw waveform samples or Mel-bins)
        self.audio_head = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, 24000)  # Predicts 1 second of 24kHz audio per step for simplicity
        )

    def forward(self, text_tokens: torch.Tensor, speaker_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_tokens: [Batch, T_phonemes]
            speaker_embedding: [Batch, 192]
        Returns:
            predicted_audio: [Batch, Audio_Samples]
            cross_attn_matrix: [Batch, M_frames, T_phonemes]
        """
        # Embed phonemes [Batch, T, d_model]
        phoneme_feats = self.phoneme_embedding(text_tokens)
        
        # Prepare speaker conditioning bias [Batch, d_model]
        spk_bias = self.speaker_proj(speaker_embedding)
        
        # Generate simulated acoustic query frames [Batch, M=10, d_model]
        batch_size = text_tokens.shape[0]
        acoustic_queries = torch.randn((batch_size, 10, self.d_model), device=text_tokens.device)
        
        # Pass through custom attention layer to intercept matrix A
        decoder_out, cross_attn_matrix = self.cross_attention_layer(
            x=acoustic_queries,
            memory=phoneme_feats,
            speaker_bias=spk_bias
        )
        
        # Project to audio waveform space [Batch, 24000]
        predicted_audio = self.audio_head(decoder_out.mean(dim=1))
        
        return predicted_audio, cross_attn_matrix
