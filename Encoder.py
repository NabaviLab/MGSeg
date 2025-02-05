import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """ Patch Embedding (PE) Layer """
    def __init__(self, img_size=1024, patch_size=7, stride=4, in_chans=3, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # Flatten and reshape
        x = self.norm(x)
        return x, H, W

class LinearEmbedding(nn.Module):
    """ Linear Embedding (LE) Layer """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        return x

class TransformerEncoderBlock(nn.Module):
    """ Transformer Encoder Block """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    """ Full Transformer Encoder with Downsampling """
    def __init__(self, img_size=1024, in_chans=3, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8]):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size=img_size, in_chans=in_chans, embed_dim=embed_dims[0])
        self.linear_embed = LinearEmbedding(embed_dims[0], embed_dims[0])
        
        self.stage1 = nn.Sequential(
            TransformerEncoderBlock(embed_dims[0], num_heads[0]),
            TransformerEncoderBlock(embed_dims[0], num_heads[0]),
            nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = nn.Sequential(
            TransformerEncoderBlock(embed_dims[1], num_heads[1]),
            TransformerEncoderBlock(embed_dims[1], num_heads[1]),
            nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1)
        )
        self.stage3 = nn.Sequential(
            TransformerEncoderBlock(embed_dims[2], num_heads[2]),
            TransformerEncoderBlock(embed_dims[2], num_heads[2]),
            nn.Conv2d(embed_dims[2], embed_dims[3], kernel_size=3, stride=2, padding=1)
        )
        self.stage4 = nn.Sequential(
            TransformerEncoderBlock(embed_dims[3], num_heads[3]),
            TransformerEncoderBlock(embed_dims[3], num_heads[3])
        )

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.linear_embed(x)
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, H, W)  # Reshape for CNN processing
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return [x1, x2, x3, x4]