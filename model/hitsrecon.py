import torch
import torch.nn as nn
from torch.nn.attention.varlen import varlen_attn
from typing import Any
import torch.nn.functional as F

def copy_if_tensor(x: Any | torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    return torch.tensor(x)

class SimpleVarlenAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 1. Combined Projection for Q, K, and V
        # Maps the input to 3x the dimension so we can slice out Q, K, and V
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        
        # 2. Output Projection to mix head information
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x_packed, cu_seq, max_len):
        """
        Args:
            x_packed: (Total_Tokens, Embed_Dim) - All hits in batch stacked.
            cu_seq:   (Batch_Size + 1) - Cumulative hit indices.
            max_len:  int - The number of hits in the largest event of the batch.
        """
        # Step 1: Project to Q, K, and V simultaneously
        qkv = self.qkv_proj(x_packed) # (Total_Tokens, 3 * Embed_Dim)
        
        # Step 2: Split into Q, K, V and reshape for heads
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to (Total_Tokens, Num_Heads, Head_Dim)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        # Step 3: Call the Variable Length Attention operator
        # This kernel uses cu_seq to ensure hits only attend to others in the same event
        attn_out = varlen_attn(
            query=q, 
            key=k, 
            value=v, 
            cu_seq_q=cu_seq, 
            cu_seq_k=cu_seq, 
            max_q=max_len, 
            max_k=max_len, 
            is_causal=False # For PMT hits, attention is usually bidirectional
        )

        # Step 4: Final mixing and projection
        # Flatten the heads back to (Total_Tokens, Embed_Dim)
        attn_out = attn_out.view(-1, self.embed_dim)
        
        return self.out_proj(attn_out)
        
class VarlenEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        # 1. Attention Block
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SimpleVarlenAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        
        # 2. Feed-Forward Block (2 Linear Layers)
        ff_dim = ff_dim or 4 * embed_dim
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.activation = nn.GELU()  
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, cu_seq, max_len):
        """
        x: Packed tensor (Total_Hits, Embed_Dim)
        cu_seq: Cumulative sequence lengths
        max_len: Max hit count in this batch
        """
        # --- Sub-Layer 1: Multi-Head Attention ---
        # Pre-LN: Norm -> Attn -> Dropout -> Add
        residual = x
        x = self.norm1(x)
        x = self.attention(x, cu_seq, max_len)
        x = self.dropout1(x)
        x = residual + x  # Residual connection 1

        # --- Sub-Layer 2: Feed-Forward Network ---
        # Pre-LN: Norm -> Linear1 -> Activation -> Linear2 -> Dropout -> Add
        residual = x
        x = self.norm2(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = residual + x  # Residual connection 2

        return x
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers=4, dropout=0.1):
        super().__init__()
        
        # We use ModuleList so PyTorch can track the parameters in each layer
        self.layers = nn.ModuleList([
            VarlenEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # The final 'Reset' for the data after all those residual additions
        #self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, cu_seq, max_len):
        # Pass the 'packed' hits through each layer sequentially
        for layer in self.layers:
            x = layer(x, cu_seq, max_len)
        
        # One last standardization before leaving the Encoder
        return x
class ReconDecoder(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int = 5):
        super().__init__()
        # Often helpful to have one hidden layer for 'thinking' 
        # before squashing to the final 4 coordinates.
        hidden_dim = embed_dim*2
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            #nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (Batch_Size, embed_dim)
        return self.net(x)
    
@torch.compile(dynamic=True, fullgraph=True)
class SNOPosEnegyRecon(nn.Module):
    def __init__(
        self, n_pmts: int, d_model: int, num_heads: int, ff_dim: int, num_layers: int = 6, dropout: float = 0.1,
        time_shift: float = 308.5,
        time_scale: float = 40.99,
        energy_shift: float = 5.25,
        energy_scale: float = 2.5,
        pos_shift: float = 0,
        pos_scale: float = 1200.0
    ):
        super().__init__()
        self.register_buffer("time_shift", torch.tensor(time_shift))
        self.register_buffer("time_scale", torch.tensor(time_scale))
        self.register_buffer("energy_shift", torch.tensor(energy_shift))
        self.register_buffer("energy_scale", torch.tensor(energy_scale))
        self.register_buffer("pos_shift", torch.tensor(pos_shift))
        self.register_buffer("pos_scale", torch.tensor(pos_scale))
        # 1. Embeddings
        #n_pmts: total pmts in SNO+ 9728
        self.pmt_id_embeddings = nn.Embedding(n_pmts, d_model)
        """
        self.hit_time_embedder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, d_model),
        )
        """
        self.hit_time_embedder = nn.Sequential(
            nn.Linear(1, d_model *2 ),
            nn.Tanh(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # 2. Encoder (The logic you provided)
        self.encoder = TransformerEncoder(d_model, num_heads, ff_dim, num_layers, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)
        
        # 3. Output Head
        self.recon_decoder = ReconDecoder(d_model, output_dim=5)
    
    # User-Defined (Global) scales and shifts
    def add_input_norm(
        self,
        input_norm: bool = True,
    ): # time_shift = 308.5 std:dev = 40.99
        self.input_norm = input_norm

    def energy_normlise(self, energy: torch.FloatTensor) -> torch.FloatTensor:
        return (energy - self.energy_shift)/self.energy_scale

    def energy_unnormlise(self, norenergy: torch.FloatTensor) -> torch.FloatTensor:
        return (norenergy * self.energy_scale) + self.energy_shift

    def time_normalise(self, evtime: torch.FloatTensor) -> torch.FloatTensor:
        return (evtime - self.time_shift)/self.time_scale
    
    def time_unnormalise(self, norevtime: torch.FloatTensor) -> torch.FloatTensor:
        return (norevtime * self.time_scale) + self.time_shift
    
    def pos_normalise(self, pos: torch.FloatTensor) -> torch.FloatTensor:
        return (pos - self.pos_shift)/self.pos_scale
    
    def pos_unnormalise(self, norpos: torch.FloatTensor) -> torch.FloatTensor:
        return (norpos * self.pos_scale) + self.pos_shift
    
    def input_normalise(self, hittime: torch.FloatTensor ) -> torch.FloatTensor:
        nor_inputs = self.time_normalise(hittime)
        #print("nor_inputs",nor_inputs)
        return nor_inputs
    
    def output_normalise(self, truth: dict[str ,torch.FloatTensor]) -> torch.FloatTensor:
        out = {}
        out["position"] = self.pos_normalise(truth["position"])
        out["energy"] = self.energy_normlise(truth["energy"])
        out["evtime"] = self.time_normalise(truth["evtime"])
        #print("outnormalise",out)
        return out
    def output_unnormalise(self, predict: dict[str ,torch.FloatTensor]) -> torch.FloatTensor:
        out = {} 
        out["position"]= self.pos_unnormalise(predict["position"])
        out["energy"] = self.energy_unnormlise(predict["energy"])
        out["evtime"] = self.time_unnormalise(predict["evtime"])
        #print("outunnormalise",out["energy"])
        return out

    def forward(self, hit_ids:torch.LongTensor , hit_times:torch.FloatTensor, cu_seq, max_len):
        # globally normalise features
        hit_times = self.input_normalise(hit_times)

        # --- Step 1: Feature Embedding ---
        # B: B events in one batch
        # n : nhits in one event 
        # d_model(d) :  embedding: normally 64
        # (N,n,d) -> varlen (SUMoverB(n),d) advanced indexing
        id_embeds = self.pmt_id_embeddings(hit_ids)
        time_embeds = self.hit_time_embedder(hit_times.unsqueeze(-1))
        
        x = id_embeds + time_embeds # Shape: # (SUMoverB(n), d)

        # --- Step 2: Transformer Encoding ---
        with torch.amp.autocast("cuda", torch.bfloat16):
            x = self.encoder(x, cu_seq, max_len)
            x = self.final_norm(x) #Shape: (SUMoverB(n), d)

        # --- Step 3: Global Pooling (Varlen safe) ---
        # (SUMoverB(n) d) (before pooling)-> (B,d ) (After pooling)
        # Using the cu_seq to identify event boundaries:
        # Sums over each n_i to give output (B, d)
        x = torch.segment_reduce(x, reduce="mean", offsets=cu_seq, axis=0)
        #x = torch.segment_reduce(x, reduce="sum", offsets=cu_seq, axis=0)
        
        # (B,5)
        out = self.recon_decoder(x)
        # --- Step 4: Final Regression ---
        # Predicts [Energy, evtime, X, Y, Z ]
        #out_dict = {"energy": F.softplus(out[..., 0]), "evtime": out[..., 1],"position": out[..., -3:]}
        out_dict = {"energy": out[..., 0], "evtime": out[..., 1],"position": out[..., -3:]}
        return out_dict