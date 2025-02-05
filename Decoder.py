import torch
import torch.nn as nn
import torch.nn.functional as F

class PFFN(nn.Module):
    """ Position-Wise Feedforward Network (PFFN) """
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.norm(x)

class Decoder(nn.Module):
    """ Cascaded CNN and FFN Decoder """
    def __init__(self, embed_dims=[64, 128, 256, 512], output_nc=2):
        super().__init__()
        
        self.ffn_layers = nn.ModuleList([
            PFFN(embed_dims[i]) for i in range(len(embed_dims))
        ])
        
        self.upsample_layers = nn.ModuleList([
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) for _ in range(len(embed_dims))
        ])
        
        self.conv_fusion = nn.Conv2d(embed_dims[0] * 4, embed_dims[0], kernel_size=1)
        
        self.final_ffn = PFFN(embed_dims[0])
        self.final_upsample = nn.ConvTranspose2d(embed_dims[0], output_nc, kernel_size=3, stride=4, padding=1, output_padding=3)

    def forward(self, fdb_outputs):
        standardized_features = [self.ffn_layers[i](fdb_outputs[i]) for i in range(len(fdb_outputs))]
        
        upsampled_features = [self.upsample_layers[i](standardized_features[i]) for i in range(len(standardized_features))]
        
        fused_features = torch.cat(upsampled_features, dim=1)
        fused_features = self.conv_fusion(fused_features)
        
        adjusted_features = self.final_ffn(fused_features)
        
        output = self.final_upsample(adjusted_features)
        return output