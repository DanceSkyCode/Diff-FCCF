from torch import nn
import torch
import numpy as np
from abc import abstractmethod
from scipy.sparse.csgraph import laplacian
from .__init__ import time_embedding
from .__init__ import Downsample
from .__init__ import Upsample
from Diffusion import GraformerLayer
import numpy as np
import torch
from scipy.sparse.csgraph import laplacian
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0, "Embedding dimension must be even"
    
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000**omega)  # shape: [embed_dim // 2]

    pos = pos.reshape(-1)  
    out = torch.einsum('p,d->pd', pos, omega)  

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # [H*W, embed_dim]

    return emb
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    import math
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')  # (H, W)

    grid = torch.stack(grid, dim=0)  # (2, H, W)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1).reshape(-1, embed_dim)  # [H*W, C]
def get_pixel_graph(B, H, W):
    num_nodes = H * W
    adj = np.zeros((B, num_nodes, num_nodes), dtype=np.float32)  # Add batch dimension

    def pixel_index(i, j):
        return i * W + j

    for b in range(B):
        for i in range(H):
            for j in range(W):
                idx = pixel_index(i, j)
                neighbors = []
                if i > 0: neighbors.append(pixel_index(i - 1, j))     
                if i < H - 1: neighbors.append(pixel_index(i + 1, j)) 
                if j > 0: neighbors.append(pixel_index(i, j - 1))     
                if j < W - 1: neighbors.append(pixel_index(i, j + 1)) 

                for n_idx in neighbors:
                    adj[b, idx, n_idx] = 1
                    adj[b, n_idx, idx] = 1  

    # Calculate the Laplacian matrix for each graph in the batch
    lap_mat = np.zeros_like(adj)
    for b in range(B):
        lap_mat[b] = laplacian(adj[b], normed=False)
    
    return adj, lap_mat

# use GN for norm layer
def group_norm(channels):
    return nn.GroupNorm(32, channels)


#  time_embedding block
class TimeBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """

        """


class TimeSequential(nn.Sequential, TimeBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimeBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# ******** Attention mudule ***********
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(AttentionBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        attention_channel = self.sigmoid(avg_out + max_out)
        attention_spatial = self.conv(x)
        attention_spatial = self.sigmoid(attention_spatial)
        attention = attention_channel * attention_spatial
        return attention * x


class ResBlock(TimeBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout, add_time):
        super().__init__()
        self.add_time = add_time
        self.conv1 = nn.Sequential(
            group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            group_norm(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class NoisePred(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 model_channels,
                 num_res_blocks,
                 dropout,
                 time_embed_dim_mult,
                 down_sample_mult,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.down_sample_mult = down_sample_mult

        # time embedding
        time_embed_dim = model_channels * time_embed_dim_mult
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        down_channels = [model_channels * i for i in down_sample_mult]
        up_channels = down_channels[::-1]

        downBlock_chanNum = [num_res_blocks + 1] * (len(down_sample_mult) - 1)
        downBlock_chanNum.append(num_res_blocks)  # [3, 3, 3, 2]
        upBlock_chanNum = downBlock_chanNum[::-1]
        self.downBlock_chanNum_cumsum = np.cumsum(downBlock_chanNum)
        self.upBlock_chanNum_cumsum = np.cumsum(upBlock_chanNum)[:-1]

        self.inBlock = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)

        # DownSample block
        self.downBlock = nn.ModuleList()
        self.attention_block = nn.ModuleList()
        self.attention_block_1 = nn.ModuleList()
        down_init_channel = model_channels
        for level, channel in enumerate(down_channels):
            # **************attention layer *******************
            attention_layer = AttentionBlock(channel)
            attention_layer_1 = AttentionBlock(channel)
            self.attention_block.append(attention_layer)
            self.attention_block_1.append(attention_layer_1)
            for _ in range(num_res_blocks):
                layer1 = ResBlock(in_channels=down_init_channel,
                                  out_channels=channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout,
                                  add_time=True)
                down_init_channel = channel
                self.downBlock.append(TimeSequential(layer1))

            if level != len(down_sample_mult) - 1:
                down_layer = Downsample(channels=channel)
                self.downBlock.append(TimeSequential(down_layer))

        # middle block
        self.middleBlock = nn.ModuleList()
        for _ in range(num_res_blocks):
            layer2 = ResBlock(in_channels=down_channels[-1],
                              out_channels=down_channels[-1],
                              time_channels=time_embed_dim,
                              dropout=dropout,
                              add_time=False)
            self.middleBlock.append(TimeSequential(layer2))

        # upsample block
        self.upBlock = nn.ModuleList()
        up_init_channel = down_channels[-1]
        for level, channel in enumerate(up_channels):
            if level == len(up_channels) - 1:
                out_channel = model_channels
            else:
                out_channel = channel // 2
            for _ in range(num_res_blocks):
                layer3 = ResBlock(in_channels=up_init_channel,
                                  out_channels=out_channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout,
                                  add_time=False)
                up_init_channel = out_channel
                self.upBlock.append(TimeSequential(layer3))
            if level > 0:
                up_layer = Upsample(channels=out_channel)
                self.upBlock.append(TimeSequential(up_layer))

        # upsample and fusion block
        self.fusionBlock = nn.ModuleList()
        up_init_channel = down_channels[-1]
        for level, channel in enumerate(up_channels):
            if level == len(up_channels) - 1:
                out_channel = model_channels
            else:
                out_channel = channel // 2
            for _ in range(num_res_blocks):
                layer4 = ResBlock(in_channels=up_init_channel,
                                  out_channels=out_channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout,
                                  add_time=False)
                up_init_channel = out_channel
                self.fusionBlock.append(TimeSequential(layer4))
            if level > 0:
                up_layer = Upsample(channels=out_channel)
                self.fusionBlock.append(TimeSequential(up_layer))

        # out block
        self.outBlock = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

        # fusion out block
        self.fusion_outBlock1 = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.fusion_outBlock2 = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.graformer = GraformerLayer(
            d_model=self.model_channels * self.down_sample_mult[-1],
            d_ff=self.model_channels * 4,
            heads=4,
            cheb_K=3
        )
        self.register_buffer('positional_embedding', get_2d_sincos_pos_embed(embed_dim=512, grid_size=32).unsqueeze(0))  # [1, 1024, 512]
    def forward(self, x, timesteps):
        embedding = time_embedding(timesteps, self.model_channels)
        time_emb = self.time_embed(embedding)

        res_noise = []
        res_fusion = []

        # in stage
        x = self.inBlock(x)

        # down stage
        h = x
        num_down = 1
        for down_block in self.downBlock:
            h = down_block(h, time_emb)
            if num_down in self.downBlock_chanNum_cumsum:
                res_noise.append(h)
                res_fusion.append(h)
            num_down += 1
        for middle_block in self.middleBlock:
            h = middle_block(h, time_emb)
        B, C, H, W = h.shape
        positional_emb = self.positional_embedding.expand(B, H*W, C)
        h = h.view(B, C, H*W).permute(0, 2, 1)  # [B, N, C] 变换为 [B, H*W, C]
        h = h + positional_emb
        adj_matrix, laplacian_matrix = get_pixel_graph(B, H, W)
        h = h.reshape(B, C, H*W).permute(0, 2, 1)  # [B, N, C]
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(h.device)
        laplacian_matrix = torch.tensor(laplacian_matrix, dtype=torch.float32).to(h.device)
        h = self.graformer(h, adj_matrix, laplacian_matrix)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        x1 = h
        x2 = h
        attention_layer = self.attention_block[3]
        attention_layer_1 = self.attention_block_1[3]
        x1 = x1 + attention_layer(res_noise.pop())
        x2 = x2 + attention_layer_1(res_fusion.pop())
        assert len(res_noise) == len(self.upBlock_chanNum_cumsum)
        assert len(res_fusion) == len(self.upBlock_chanNum_cumsum)
        # up stage
        num_up = 1
        num_attention = 2
        for up_block in self.upBlock:
            if num_up in self.upBlock_chanNum_cumsum:  # [2,5,8]
                x1 = up_block(x1, time_emb)
                x1_crop = x1[:, :, :res_noise[-1].shape[2], :res_noise[-1].shape[3]]
                attention_layer = self.attention_block[num_attention]
                num_attention = num_attention - 1
                x1 = x1_crop + attention_layer(res_noise.pop())
                # x1 = x1_crop + res_noise.pop()
            else:
                x1 = up_block(x1, time_emb)
            num_up += 1
        assert len(res_noise) == 0
        # # fusion stage
        num_up = 1
        num_attention = 2
        for fusion_block in self.fusionBlock:

            if num_up in self.upBlock_chanNum_cumsum:  # [2,5,8]
                x2 = fusion_block(x2, time_emb)
                x2_crop = x2[:, :, :res_fusion[-1].shape[2], :res_fusion[-1].shape[3]]
                attention_layer_1 = self.attention_block_1[num_attention]
                num_attention = num_attention - 1
                x2 = x2_crop + attention_layer_1(res_fusion.pop())
                # x2 = x2_crop + res_fusion.pop()
            else:
                x2 = fusion_block(x2, time_emb)
            num_up += 1
        assert len(res_fusion) == 0
        # out stage
        noise_out = self.outBlock(x1)
        fusion_out1= self.fusion_outBlock1(x2)
        fusion_out2 = self.fusion_outBlock2(x2)
        DVField1 = self.fusion_outBlock1(x2).permute(0, 2, 3, 1)
        DVField2 = self.fusion_outBlock2(x2).permute(0, 2, 3, 1)
        DVField = torch.cat( (DVField1,DVField2),dim=3)
        return noise_out, fusion_out1, fusion_out2, DVField