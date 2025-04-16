import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum, auto
from dataclasses import dataclass

# ------------------------
# Configurations and Enums
# ------------------------

class PositionalEncodings(Enum):
    SINUSOID = auto()
    LEARNED = auto()

@dataclass
class TransformerConfig:
    # Hyperparameters used in the Transformer architectures
    seed: int = 1
    vocab_size: int = None
    output_size: int = None
    embedding_dim: int = 64
    num_layers: int = 4
    num_heads: int = 8
    use_causal_mask: bool = True
    emb_init_scale: float = 0.02
    pos_encodings: PositionalEncodings = PositionalEncodings.SINUSOID
    max_sequence_length: int = None
    widening_factor: int = 4
    apply_qk_layernorm: bool = False
    apply_post_ln: bool = True

    def __post_init__(self):
        if self.output_size is None:
            self.output_size = self.vocab_size

# --------------------
# Multi-Head Attention
# --------------------

class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, num_heads: int, num_hiddens_per_head: int, embedding_size: int, apply_qk_layernorm: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.num_hiddens_per_head = num_hiddens_per_head
        self.apply_qk_layernorm = apply_qk_layernorm
        self.total_dim = num_heads * num_hiddens_per_head

        self.q_linear = nn.Linear(embedding_size, self.total_dim, bias=False)
        self.k_linear = nn.Linear(embedding_size, self.total_dim, bias=False)
        self.v_linear = nn.Linear(embedding_size, self.total_dim, bias=False)
        self.out_linear = nn.Linear(self.total_dim, embedding_size, bias=False)

        if apply_qk_layernorm:
            self.q_layernorm = nn.LayerNorm(self.total_dim)
            self.k_layernorm = nn.LayerNorm(self.total_dim)
        else:
            self.q_layernorm = None
            self.k_layernorm = None

    def forward(self, inputs_q: torch.Tensor, inputs_kv: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # inputs_q: (B, T, embedding_size)
        q = self.q_linear(inputs_q)
        k = self.k_linear(inputs_kv)
        v = self.v_linear(inputs_kv)

        if self.apply_qk_layernorm:
            q = self.q_layernorm(q)
            k = self.k_layernorm(k)

        B, T, _ = q.shape
        # Reshape into (B, num_heads, T, num_hiddens_per_head)
        q = q.view(B, T, self.num_heads, self.num_hiddens_per_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.num_hiddens_per_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.num_hiddens_per_head).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.einsum('bhqd, bhkd -> bhqk', q, k)
        scale = math.sqrt(self.num_hiddens_per_head)
        attn_scores = attn_scores / scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.einsum('bhqk, bhkd -> bhqd', attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.total_dim)
        return self.out_linear(out)

# --------------------
# Positional Encodings
# --------------------

def sinusoid_position_encoding(sequence_length: int, hidden_size: int, max_timescale: float = 1e4) -> torch.Tensor:
    pos = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(0, hidden_size, 2, dtype=torch.float32)
    inv_freq = max_timescale ** (-dim / hidden_size)
    sinusoid_inp = pos * inv_freq.unsqueeze(0)
    pe = torch.zeros(sequence_length, hidden_size)
    pe[:, 0::2] = torch.sin(sinusoid_inp)
    pe[:, 1::2] = torch.cos(sinusoid_inp)
    return pe

# -----------------------------------
# Embedding with Positional Encodings
# -----------------------------------

class EmbedSequences(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.embedding_dim)

        if config.pos_encodings == PositionalEncodings.LEARNED:
            assert config.max_sequence_length is not None
            self.pos_emb = nn.Embedding(config.max_sequence_length, config.embedding_dim)
        else:
            self.pos_emb = None
        nn.init.trunc_normal_(self.token_emb.weight, std=config.emb_init_scale)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        # Sequences: (B, T)
        embeddings = self.token_emb(sequences)  # (B, T, embedding_dim)
        embeddings = embeddings * math.sqrt(self.config.embedding_dim)
        B, T, D = embeddings.shape

        if self.config.pos_encodings == PositionalEncodings.SINUSOID:
            pos_encoding = sinusoid_position_encoding(T, D)
            pos_encoding = pos_encoding.to(embeddings.device)
        else:  # LEARNED
            positions = torch.arange(T, device=embeddings.device).unsqueeze(0)
            pos_encoding = self.pos_emb(positions)  # (1, T, D)

        return embeddings + pos_encoding

# --------------------
# Utility: Shift Right
# --------------------

def shift_right(sequences: torch.Tensor) -> torch.Tensor:
    bos = torch.zeros(sequences.size(0), 1, dtype=sequences.dtype, device=sequences.device)
    padded = torch.cat([bos, sequences], dim=1)
    return padded[:, :-1]

# ---------------------
# MLP/Feedforward Block
# ---------------------

class MLPBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        ffn_dim = config.embedding_dim * config.widening_factor
        self.fc1 = nn.Linear(config.embedding_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(config.embedding_dim, ffn_dim, bias=False)
        self.fc_out = nn.Linear(ffn_dim, config.embedding_dim, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split1 = self.fc1(x)
        split2 = self.fc2(x)
        gate_output = self.activation(split1) * split2
        return self.fc_out(gate_output)

# -----------------
# Transformer Layer
# -----------------

class TransformerLayer(nn.Module):
    """A single Transformer layer containing attention and MLP blocks with residual connections"""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.attn_ln = nn.LayerNorm(config.embedding_dim)
        self.attention = MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            num_hiddens_per_head=config.embedding_dim // config.num_heads,
            embedding_size=config.embedding_dim,
            apply_qk_layernorm=config.apply_qk_layernorm,
        )
        self.mlp_ln = nn.LayerNorm(config.embedding_dim)
        self.mlp = MLPBlock(config)
        self.use_causal_mask = config.use_causal_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block with residual connection
        residual = x
        x_norm = self.attn_ln(x)
        if self.use_causal_mask:
            B, T, _ = x.shape
            mask = torch.tril(torch.ones((T, T), device=x.device)).unsqueeze(0).unsqueeze(0)
            mask = mask.expand(B, 1, T, T)
        else:
            mask = None
        attn_out = self.attention(x_norm, x_norm, mask)
        x = residual + attn_out

        # MLP block with residual connection.
        residual = x
        x_norm = self.mlp_ln(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out
        return x

# -------------------
# Transformer Decoder
# -------------------

class TransformerDecoder(nn.Module):
    """The complete Transformer decoder with input embedding, multiple layers, and output projection"""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_sequences = EmbedSequences(config)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.post_ln = nn.LayerNorm(config.embedding_dim) if config.apply_post_ln else None
        self.output_linear = nn.Linear(config.embedding_dim, config.output_size)

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        # Targets: (B, T)
        inputs = shift_right(targets)
        x = self.embed_sequences(inputs)
        for layer in self.layers:
            x = layer(x)
        if self.post_ln is not None:
            x = self.post_ln(x)
        logits = self.output_linear(x)  # (B, T, output_size)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

# -----------------
# Predictor Wrapper
# -----------------

class Predictor:
    def __init__(self, model: nn.Module):
        self.model = model

    def predict(self, targets: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(targets)

def build_transformer_predictor(config: TransformerConfig) -> Predictor:
    model = TransformerDecoder(config)
    return Predictor(model)