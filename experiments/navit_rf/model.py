import torch
import torch.nn as nn

# -----------------------
# Utilities
# -----------------------

class LearnedTimeEmbedding(nn.Module):
    """
    Learnable embedding that maps scalar t in [0,1] to a d_model vector.
    """
    def __init__(self, d_model: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        in_dim = 1
        for _ in range(max(depth - 1, 1)):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.SiLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, d_model))
        self.net = nn.Sequential(*layers)

    def forward(self, t):  # t: (B,)
        return self.net(t.view(t.shape[0], 1))


class PatchEmbed(nn.Module):
    """
    Simple ViT-style patch embedding using Conv2d (stride=patch).
    """
    def __init__(self, in_ch=3, d_model=512, patch=16):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)

    def forward(self, x):  # x: (B, C, H, W)
        x = self.proj(x)   # (B, d_model, H/patch, W/patch)
        B, D, Ht, Wt = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D) where N = Ht*Wt
        return x, (Ht, Wt)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_head=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class ViTVelocity(nn.Module):
    """
    ViT encoder that predicts per-pixel velocity field v(x_t, t).
    - Input: x_t (B,C,H,W), scalar t (B,)
    - Output: velocity (B,C,H,W) same size as input
    """
    def __init__(
        self,
        img_size=256,
        patch=16,
        in_ch=3,
        d_model=512,
        depth=8,
        n_head=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        assert img_size % patch == 0, "img_size must be divisible by patch size"
        self.img_size = img_size
        self.patch = patch
        self.in_ch = in_ch
        self.d_model = d_model

        self.patch_embed = PatchEmbed(in_ch=in_ch, d_model=d_model, patch=patch)
        num_patches = (img_size // patch) * (img_size // patch)

        # learned 2D positional embeddings for patches
        self.pos = nn.Parameter(torch.zeros(1, num_patches, d_model))

        # time conditioning
        self.time_embed = LearnedTimeEmbedding(d_model=d_model, hidden=512, depth=3)

        # transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_head=n_head, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

        # project tokens back to patch pixels and unpatchify
        # each token -> (patch*patch*in_ch) values
        self.to_patch_pixels = nn.Linear(d_model, in_ch * patch * patch)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def unpatchify(self, x_tokens, hw):
        """
        x_tokens: (B, N, in_ch*patch*patch)
        hw: (Ht, Wt) where Ht=H/patch, Wt=W/patch
        Returns: (B, in_ch, H, W)
        """
        B, N, PP = x_tokens.shape
        Ht, Wt = hw
        p = self.patch
        x = x_tokens.view(B, Ht, Wt, self.in_ch, p, p)  # (B,Ht,Wt,C,p,p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()    # (B,C,Ht,p,Wt,p)
        x = x.view(B, self.in_ch, Ht * p, Wt * p)       # (B,C,H,W)
        return x

    def forward(self, x, t):  # x: (B,C,H,W), t: (B,)
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "resize/crop to img_size"

        # Patchify
        tokens, hw = self.patch_embed(x)              # (B, N, D), hw=(Ht,Wt)
        N = tokens.size(1)

        # Add positional enc
        tokens = tokens + self.pos[:, :N, :]

        # Time conditioning: add to every token
        t_vec = self.time_embed(t)                    # (B, D)
        tokens = tokens + t_vec.unsqueeze(1)         # broadcast over tokens

        # Transformer
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        # Predict per-patch pixels, then unpatchify
        patch_pixels = self.to_patch_pixels(tokens)   # (B, N, C*p*p)
        v = self.unpatchify(patch_pixels, hw)         # (B, C, H, W)
        return v
