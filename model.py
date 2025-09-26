import torch
import torch.nn as nn
import torch.nn.functional as F

# a residual 3D convolution block
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, dropout_probability):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout3d(dropout_probability) if dropout else nn.Identity()

    def forward(self, x):
        # the goal of this block is not to learn a mapping x -> y, but to learn a mapping F(x) that allows x + F(x) = y
        # in other words, how should I tweak x by parameterizing F(x) and adding it to x - in order to approximate y
        x_conv = self.block(x)
        x_res = self.residual(x)

        return self.dropout(x_conv + x_res)

# a 3D attention gate module
class AttentionGate3D(nn.Module):
    def __init__(self, in_channels_enc, in_channels_dec, latent_channels):
        super().__init__()
        self.W_enc = nn.Sequential(
            nn.Conv3d(in_channels_enc, latent_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(latent_channels)
        )

        self.W_dec = nn.Sequential(
            nn.Conv3d(in_channels_dec, latent_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(latent_channels)
        )

        self.attention = nn.Sequential(
            nn.Conv3d(latent_channels, 1, kernel_size=1, bias=True), # gives us spatial attention scores
            nn.Sigmoid()
        )

        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, enc, dec):
        enc_latent = self.W_enc(enc)
        dec_latent = self.W_dec(dec)
        att_scores = self.attention(self.leaky_relu(enc_latent + dec_latent))
        return enc * att_scores

# a 3D cross attention module
class CrossAttention3D(nn.Module):
    def __init__(self, in_channels_enc, in_channels_dec, latent_channels, num_heads, dropout, dropout_probability):
        super().__init__()
        if latent_channels % num_heads != 0:
            raise ValueError(f"latent_channels must be divisible by num_heads")

        self.lc = latent_channels
        self.n_h = num_heads
        self.n_lc_per_h = latent_channels // num_heads

        # projections to token latent space
        self.q_proj = nn.Conv3d(in_channels_dec, latent_channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv3d(in_channels_enc, latent_channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv3d(in_channels_enc, latent_channels, kernel_size=1, bias=False)

        self.norm_q = nn.LayerNorm(latent_channels)
        self.norm_k = nn.LayerNorm(latent_channels)
        self.norm_v = nn.LayerNorm(latent_channels)

        self.out_proj = nn.Conv3d(latent_channels, in_channels_dec, kernel_size=1, bias=False)
        
        self.proj_dropout = nn.Dropout3d(dropout_probability) if dropout else nn.Identity()
        self.att_dropout = nn.Dropout(dropout_probability) if dropout else nn.Identity()
    
    # flatten spatial dimensions, (b, c, d, h, w) to (b, d * h * w, c)
    def _to_tokens(self, x):
        b, c, d, h, w = x.shape
        tokens = x.view(b, c, d * h * w).transpose(1, 2)
        return tokens, (b, c, d, h, w)
    
    # reconstruct spatial dimensions, (b, d * h * w, lc) to (b, lc, d, h, w)
    def _from_tokens(self, x, shape):
        b, _, d, h, w = shape
        rec_spatial = x.transpose(1, 2).view(b, self.lc, d, h, w)
        return rec_spatial
    
    # split attention heads, (b, d * h * w, n_h, n_lc_per_h) to (b, n_h, d * h * w, n_lc_per_h)
    def _split_heads(self, x):
        b, dhw, _ = x.shape
        att_heads = x.view(b, dhw, self.n_h, self.n_lc_per_h).transpose(1, 2)
        return att_heads

    def forward(self, enc, dec):
        q = self.q_proj(dec)
        k = self.k_proj(enc)
        v = self.v_proj(enc)

        q_t, _ = self._to_tokens(q)
        k_t, _ = self._to_tokens(k)
        v_t, _ = self._to_tokens(v)

        q_t = self.norm_q(q_t)
        k_t = self.norm_k(k_t)
        v_t = self.norm_v(v_t)

        q_h = self._split_heads(q_t)
        k_h = self._split_heads(k_t)
        v_h = self._split_heads(v_t)

        # scaled dot-product attention
        att_raw = (q_h @ k_h.transpose(-2, -1)) / (self.n_lc_per_h ** 0.5)
        att_scores = self.att_dropout(att_raw.softmax(dim=-1))
        att_ctx = att_scores @ v_h # (b, n_h, d * h * w, n_lc_per_h)
        
        # merge attention heads
        att_ctx = att_ctx.transpose(1, 2).contiguous().view(q_t.shape[0], q_t.shape[1], self.lc) # (b, d * h * w, lc)

        # reconstruct spatial feature map
        att_ctx_map = self._from_tokens(att_ctx, (dec.shape[0], self.lc, dec.shape[2], dec.shape[3], dec.shape[4]))
        att_ctx_map = self.proj_dropout(self.out_proj(att_ctx_map)) # (b, in_channels_dec, d, h, w)

        return dec + att_ctx_map

# a 3D deconvolution and skip block with an attention gate
class UpBlock3D_AG(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, dropout, dropout_probability):
        super().__init__()
        self.att_gate = AttentionGate3D(in_channels, in_channels, latent_channels)
        self.conv = ConvBlock3D(in_channels * 2, in_channels, dropout=dropout, dropout_probability=dropout_probability)
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2) # deconvolution (upsampling)

    def forward(self, x, skip=None):
        if skip is None:
            return self.up(x)

        # padding for size mismatches
        diffX = skip.size(4) - x.size(4)
        diffY = skip.size(3) - x.size(3)
        diffZ = skip.size(2) - x.size(2)

        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2,
                                  diffZ // 2, diffZ - diffZ // 2])
        
        enc_ctx = self.att_gate(skip, x) # apply attention gating to encoder features
        x = torch.cat((enc_ctx, x), dim=1) # concatenate contextualized encoder and decoder features
        x = self.conv(x) # learn a transformation that condenses them

        return self.up(x)

# a 3D deconvolution and skip block with cross attention
class UpBlock3D_CA(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, num_heads, dropout, dropout_probability):
        super().__init__()
        self.cross_att = CrossAttention3D(
            in_channels, in_channels, latent_channels,
            num_heads=num_heads, dropout=dropout, dropout_probability=dropout_probability
        )

        self.conv = ConvBlock3D(in_channels * 2, in_channels, dropout=dropout, dropout_probability=dropout_probability)
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2) # deconvolution (upsampling)

    def forward(self, x, skip=None):
        if skip is None:
            return self.up(x)

        # padding for size mismatches
        diffX = skip.size(4) - x.size(4)
        diffY = skip.size(3) - x.size(3)
        diffZ = skip.size(2) - x.size(2)

        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2,
                                  diffZ // 2, diffZ - diffZ // 2])
        
        dec_ctx = self.cross_att(skip, x) # apply cross attention to decoder features
        x = torch.cat((skip, dec_ctx), dim=1) # concatenate encoder and contextualized decoder features
        x = self.conv(x)

        return self.up(x)

# a residual 3D unet with multiple types of attention
class ResAtt3DUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, num_filters=32, num_heads=2, dropout=False, dropout_probability=0.2):
        super().__init__()
        f = num_filters

        # encoder blocks
        self.enc0 = ConvBlock3D(in_channels, f, dropout=dropout, dropout_probability=dropout_probability)
        self.enc1 = ConvBlock3D(f, f * 2, dropout=dropout, dropout_probability=dropout_probability)
        self.enc2 = ConvBlock3D(f * 2, f * 4, dropout=dropout, dropout_probability=dropout_probability)
        self.enc3 = ConvBlock3D(f * 4, f * 8, dropout=dropout, dropout_probability=dropout_probability)

        self.bottleneck = ConvBlock3D(f * 8, f * 16, dropout=dropout, dropout_probability=dropout_probability) # deepest feature map

        self.pool = nn.MaxPool3d(2) # pooling layer (downsampling)

        # decoder blocks
        self.dec3 = UpBlock3D_CA(f * 16, f * 8, f * 16, num_heads=num_heads, dropout=dropout, dropout_probability=dropout_probability)
        self.dec2 = UpBlock3D_AG(f * 8, f * 4, f * 8, dropout=dropout, dropout_probability=dropout_probability)
        self.dec1 = UpBlock3D_AG(f * 4, f * 2, f * 4, dropout=dropout, dropout_probability=dropout_probability)
        self.dec0 = UpBlock3D_AG(f * 2, f, f * 2, dropout=dropout, dropout_probability=dropout_probability)
        
        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # encoder path
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # deepest feature map
        b = self.bottleneck(self.pool(e3))

        # decoder path
        d3 = self.dec3(b)
        d2 = self.dec2(d3, e3)
        d1 = self.dec1(d2, e2)
        d0 = self.dec0(d1, e1)

        # padding for size mismatches
        diffX = e0.size(4) - d0.size(4)
        diffY = e0.size(3) - d0.size(3)
        diffZ = e0.size(2) - d0.size(2)
        
        d0 = nn.functional.pad(d0, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2])
                
        return self.final_conv(d0)