import torch
import torch.nn as nn
from einops import repeat

class PosMLP(nn.Module):
    def __init__(self, in_ch=3, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
        )
    def forward(self, x):  # (B,N,3)
        return self.net(x)

class GlobalAttn(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn  = nn.Sequential(nn.Linear(dim, dim*4), nn.ReLU(), nn.Linear(dim*4, dim))
    def forward(self, x):  # (B,N,C)
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x

class SeedFormerLite(nn.Module):
    def __init__(self, k_seeds=512, up_ratio=4, dim=128, heads=4):
        super().__init__()
        self.k = k_seeds
        self.r = up_ratio
        self.embed = PosMLP(3, dim)
        self.encoder = nn.Sequential(GlobalAttn(dim, heads),
                                     GlobalAttn(dim, heads))
        # learnable seed queries
        self.query_feat = nn.Parameter(torch.randn(1, k_seeds, dim))
        # cross attention module (queries attend to encoded features)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        # heads
        self.coarse_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 3))
        self.upsample_feat = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.offset_head = nn.Sequential(nn.Linear(dim+3, dim), nn.ReLU(), nn.Linear(dim, 3))

    def forward(self, partial):  # (B,N,3)
        B = partial.size(0)
        x = self.embed(partial)           # (B,N,C)
        x = self.encoder(x)               # (B,N,C)

        q = self.query_feat.expand(B, -1, -1)  # (B,K,C)
        q2, _ = self.cross_attn(q, x, x)       # (B,K,C)
        coarse = self.coarse_head(q2)          # (B,K,3)

        # upsample: replicate each coarse point r times and learn offsets
        feat_up = self.upsample_feat(q2)                       # (B,K,C)
        feat_up = repeat(feat_up, 'b k c -> b (k r) c', r=self.r)  # (B,K*r,C)
        coarse_rep = repeat(coarse, 'b k d -> b (k r) d', r=self.r)
        noise = torch.randn_like(coarse_rep) * 0.01
        fine = self.offset_head(torch.cat([feat_up, coarse_rep], dim=-1)) + coarse_rep + noise
        return coarse, fine
