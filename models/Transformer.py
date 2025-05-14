import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, feedforward_dim=256, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        
        self.d_model = d_model
        # 多头注意力层
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, vs, eq):
        
        # 转换形状以适应多头注意力层的输入要求
        # vs = vs.transpose(0, 1)  # (seq_len, batch_size, d_model)
        # eq = eq.transpose(0, 1)  # (seq_len, batch_size, d_model)
        
        q = eq
        k,v = vs,vs 
        # 计算交叉注意力
        attn_output, attn_weights = self.attention(q, k, v)
        
        # 残差连接 + 层归一化
        attn_output = self.norm1(attn_output + eq)
        
        # 前馈网络
        ffn_output = self.ffn(attn_output)
        
        # 残差连接 + 层归一化
        output = self.norm2(ffn_output + attn_output)
        
        return output, attn_weights

class MultiLayerCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, input_dim=64,feedforward_dim=256, dropout=0.1):
        super(MultiLayerCrossAttention, self).__init__()
        self.d_model = d_model
        # 多层交叉注意力
        self.eq_input_embedding = nn.Linear(input_dim, d_model)
        self.vs_input_embedding = nn.Linear(input_dim, d_model)
        self.output_projection = nn.Linear(d_model, input_dim)
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(d_model, n_heads, feedforward_dim, dropout) for _ in range(num_layers)]
        )
    
    def forward(self, vs, eq, ts):
        # print(vs.shape,eq.shape,ts.shape)
        # print(eq.max(),eq.min(),eq.mean())
        vs = vs.transpose(0, 1)  # (seq_len, batch_size, d_model)
        eq = eq.transpose(0, 1)  # (seq_len, batch_size, d_model)
        ts.unsqueeze(-1)  # (seq_len, 1)
        time_emb = timestep_embedding(ts, dim=self.d_model)
        time_emb = time_emb.unsqueeze(0).repeat(eq.shape[0], 1, 1)
        eq = self.eq_input_embedding(eq)
        vs = self.vs_input_embedding(vs)
        eq = eq + time_emb
        # 逐层传递输入，得到多层的输出
        attn_weights_all_layers = []
        
        for layer in self.layers:
            eq, _ = layer(vs, eq)  # eq ===> (seq_len, batch_size, d_model)
            attn_weights_all_layers.append(eq)  # 将每一层的输出存储
        eq = self.output_projection(eq)
        eq = eq.transpose(0, 1)
        return eq






if __name__ == '__main__':
    # 示例使用
    batch_size = 88
    seq_len = 5
    d_model = 64
    n_heads = 8
    num_layers = 3  # 多层交叉注意力

    # 假设我们有以下输入
    vs = torch.randn(batch_size, seq_len, 64)  # 物品嵌入
    eq = torch.randn(batch_size,seq_len, 64)  # 查询嵌入
    ts = torch.randn(88)  # 时间步长
    multi_layer_attention = MultiLayerCrossAttention(d_model=d_model, n_heads=n_heads, num_layers=num_layers)

    # 获取多层交叉注意力输出
    eq_reconstructed = multi_layer_attention(vs, eq, ts)

    # 输出形状
    print(f"eq_reconstructed shape: {eq_reconstructed.shape}")  # (batch_size, seq_len, d_model)
    # print(f"attn_weights_all_layers length: {len(attn_weights_all_layers)}")  # 应该为 num_layers
