'''
Contains classes for cross-attention mechanism.
'''

import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

torch.set_num_threads(4)


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=1, dim_head=None, dropout=0.):
        super().__init__()
        if dim_head is None:
            dim_head = dim
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv_x = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv_y = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        qkv_x = self.to_qkv_x(x).chunk(3, dim=-1)
        qkv_y = self.to_qkv_y(y).chunk(3, dim=-1)
        q_x, _, _ = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_x)
        _, k_y, v_y = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_y)

        dots = torch.matmul(q_x, k_y.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v_y)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, input_size, num_filters=192, heads=1, ch_patch_size=1, num_patches=4, dim=None, dim_head=None,
                 dropout=0.):
        super().__init__()

        assert num_filters % ch_patch_size == 0, 'num_filters must be divisible by the patch size.'
        self.patch_size = [None] * 3
        self.patch_size[0] = ch_patch_size
        self.patch_size[1] = input_size[0] // num_patches
        self.patch_size[2] = input_size[1] // num_patches
        patch_dim = self.patch_size[1] * self.patch_size[2] * ch_patch_size
        if dim is None:
            dim = patch_dim

        self.to_patch_embedding_x = nn.Sequential(
            Rearrange('b (c p0) (h p1) (w p2) -> b (c h w) (p0 p1 p2)', p0=ch_patch_size,
                      p1=self.patch_size[1], p2=self.patch_size[2]),
            nn.Linear(patch_dim, dim),
        )
        self.to_patch_embedding_y = nn.Sequential(
            Rearrange('b (c p0) (h p1) (w p2) -> b (c h w) (p0 p1 p2)', p0=ch_patch_size,
                      p1=self.patch_size[1], p2=self.patch_size[2]),
            nn.Linear(patch_dim, dim),
        )
        self.unpack_embedding_y = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (c h w) (p0 p1 p2) -> b (c p0) (h p1) (w p2)', h=num_patches, w=num_patches,
                      p0=ch_patch_size, p1=self.patch_size[1],
                      p2=self.patch_size[2]),
        )

        self.norm = nn.LayerNorm(dim)
        self.attn = AttentionBlock(dim, heads, dim_head, dropout)

    def forward(self, x, y):
        x_emb = self.to_patch_embedding_x(x)
        y_emb = self.to_patch_embedding_y(y)
        x_norm = self.norm(x_emb)
        y_norm = self.norm(y_emb)

        y = self.attn(x_norm, y_norm)
        y = self.unpack_embedding_y(y)

        x = torch.cat((x, y), 1)

        return x


if __name__ == '__main__':
    x, y = torch.randn(1, 192, 64, 64), torch.randn(1, 192, 64, 64)
    net = CrossAttention(input_size=x.shape[2:], num_filters=x.shape[1], dim=512, num_patches=8)
    print(net(x, y).shape)
