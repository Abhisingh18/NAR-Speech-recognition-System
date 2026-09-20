import torch
import torch.nn as nn
import pdb


class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.llm_dim)
        # print(f"layer2 default dtype: {print(torch.get_default_dtype()) }")

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class EncoderProjectorCov1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.conv1d = nn.Conv1d(in_channels=self.encoder_dim, out_channels=self.encoder_dim, kernel_size=self.k, stride=self.k, padding=0)
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x

class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = config.qformer_layers

        self.query_len = int(config.get("query_len", 64))
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        
        return query_proj


class ResidualConv1dBlock(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=1, padding=padding)
        self.act = nn.GELU()   # or nn.ReLU()
        # Optional: use a small conv-based bottleneck if you want
    def forward(self, x):
        # x: B x C x T
        out = self.conv(x)
        out = self.act(out)
        return x + out

class EncoderProjectorConv1dResidual(nn.Module):
    def __init__(self, config, num_res_blocks: int = 1, dropout: float = 0.1):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        # Downsampling convolution (stride = k)
        self.down_conv = nn.Conv1d(
            in_channels=self.encoder_dim,
            out_channels=self.encoder_dim,
            kernel_size=self.k,
            stride=self.k,
            padding=0,  # downsample; output length = input_len // k (floor)
        )

        # Residual Conv stack (keeps same channels and length)
        self.res_blocks = nn.ModuleList(
            [ResidualConv1dBlock(self.encoder_dim, kernel_size=3) for _ in range(num_res_blocks)]
        )

        # Normalize across features (apply after transpose back to B x T x C)
        self.post_norm = nn.LayerNorm(self.encoder_dim)

        # MLP projector (Linear -> Act -> Linear -> [no final activation])
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.act1 = nn.GELU()   # or nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.linear2 = nn.Linear(2048, self.llm_dim)
        # NO activation after projection: many systems prefer raw projected embeddings

    def forward(self, x):
        # x: B x T x C (encoder_outs)
        x = x.transpose(1, 2)         # B x C x T
        x = self.down_conv(x)         # downsampled: B x C x T'
        for blk in self.res_blocks:
            x = blk(x)                # residual conv blocks
        x = x.transpose(1, 2)         # B x T' x C
        x = self.post_norm(x)         # LayerNorm on features
        # MLP projection: Linear -> Activation -> Dropout -> Linear
        x = self.linear1(x)           # B x T' x 2048
        x = self.act1(x)
        x = self.dropout(x)
        x = self.linear2(x)           # B x T' x llm_dim
        return x



class EncoderDownsamplerCov1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.conv1 = nn.Conv1d(
            in_channels=self.encoder_dim,
            out_channels=self.encoder_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=self.encoder_dim,
            out_channels=self.encoder_dim,
            kernel_size=self.k,
            stride=self.k,
            padding=0
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        return x

class EncoderProjectorLinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
    
    def forward(self, x):
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x