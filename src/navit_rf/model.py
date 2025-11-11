from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .embeddings import FourierTimeEmbedding, FractionalFourierPositionalEmbedding


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, d_model=512, patch=16):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        B, D, Ht, Wt = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        return tokens, (Ht, Wt)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_head=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.norm1(x)
        attn_out = self.attn(
            qkv,
            qkv,
            qkv,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )[0]
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTVelocity(nn.Module):
    """
    ViT backbone that predicts the rectified-flow velocity for packed NaViT batches.
    """

    def __init__(
        self,
        patch=16,
        in_ch=3,
        d_model=512,
        depth=8,
        n_head=8,
        mlp_ratio=4.0,
        time_embed: Optional[nn.Module] = None,
        pos_embed: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.patch = patch
        self.in_ch = in_ch
        self.d_model = d_model

        self.patch_embed = PatchEmbed(in_ch=in_ch, d_model=d_model, patch=patch)
        self.time_embed = time_embed or FourierTimeEmbedding(d_model=d_model)
        self.pos_embed = pos_embed or FractionalFourierPositionalEmbedding(dim=d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, n_head=n_head, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.to_patch_pixels = nn.Linear(d_model, in_ch * patch * patch)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def unpatchify(self, x_tokens: torch.Tensor, hw: Sequence[int]) -> torch.Tensor:
        B, N, _ = x_tokens.shape
        Ht, Wt = hw
        p = self.patch
        x = x_tokens.view(B, Ht, Wt, self.in_ch, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, self.in_ch, Ht * p, Wt * p)

    def _scatter_back(
        self,
        packed_pixels: torch.Tensor,
        token_spans: List[List[tuple]],
        patch_hw: torch.Tensor,
        Ht_full: int,
        Wt_full: int,
    ) -> torch.Tensor:
        recon = packed_pixels.new_zeros(
            len(patch_hw),
            self.in_ch,
            Ht_full * self.patch,
            Wt_full * self.patch,
        )
        for pack_idx, spans in enumerate(token_spans):
            for idx, start, length in spans:
                ph = int(patch_hw[idx, 0].item())
                pw = int(patch_hw[idx, 1].item())
                token_chunk = packed_pixels[pack_idx, start : start + length, :]
                v = self.unpatchify(token_chunk.unsqueeze(0), (ph, pw))[0]
                recon[idx, :, : ph * self.patch, : pw * self.patch] = v
        return recon

    def forward(
        self,
        images: torch.Tensor,
        t: torch.Tensor,
        *,
        patch_hw: torch.Tensor,
        packs: List[List[int]],
        orig_hw: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = images.device
        img_tokens, (Ht_full, Wt_full) = self.patch_embed(images)
        M = images.size(0)

        t_embed = self.time_embed(t.to(device))
        per_image_tokens: List[torch.Tensor] = []
        pos_cache: dict[tuple[int, int], torch.Tensor] = {}

        for i in range(M):
            ph = int(patch_hw[i, 0].item())
            pw = int(patch_hw[i, 1].item())
            key = (ph, pw)
            if key not in pos_cache:
                pos_cache[key] = self.pos_embed(ph, pw, device=device, dtype=img_tokens.dtype)

            mask = torch.zeros((Ht_full, Wt_full), dtype=torch.bool, device=device)
            mask[:ph, :pw] = True
            mask = mask.view(-1)
            tokens_i = img_tokens[i, mask, :]
            tokens_i = tokens_i + pos_cache[key] + t_embed[i].unsqueeze(0)
            per_image_tokens.append(tokens_i)

        seq_lengths = [
            sum(per_image_tokens[idx].shape[0] for idx in pack) for pack in packs
        ]
        max_len = max(seq_lengths)
        packed = images.new_zeros(len(packs), max_len, self.d_model)
        pad_mask = torch.ones(len(packs), max_len, dtype=torch.bool, device=device)
        token_spans: List[List[tuple]] = [[] for _ in packs]
        offsets = [0] * len(packs)

        for pack_idx, img_indices in enumerate(packs):
            for idx in img_indices:
                tokens_i = per_image_tokens[idx]
                start = offsets[pack_idx]
                length = tokens_i.shape[0]
                packed[pack_idx, start : start + length] = tokens_i
                pad_mask[pack_idx, start : start + length] = False
                token_spans[pack_idx].append((idx, start, length))
                offsets[pack_idx] += length

        for blk in self.blocks:
            packed = blk(packed, key_padding_mask=pad_mask, attn_mask=attn_mask)
        packed = self.norm(packed)
        packed_pixels = self.to_patch_pixels(packed)

        recon = self._scatter_back(packed_pixels, token_spans, patch_hw, Ht_full, Wt_full)

        if orig_hw is not None:
            for idx, (h, w) in enumerate(orig_hw.tolist()):
                recon[idx, :, h:, :] = 0
                recon[idx, :, :, w:] = 0

        return recon

