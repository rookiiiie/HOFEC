import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


class Transformer(nn.Module):
    def __init__(self, inp_res=32, dim=256, depth=1, num_heads=1, mlp_ratio=4.,drop=0.6,injection = False):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,drop=drop,injection=injection))
        
    def forward(self, query, key):
        output = query
        #print('trans')
        for i, layer in enumerate(self.layers):
            #print('trans')
            output = layer(query=output, key=key)
        # if self.injection: # FIT
        #     output = torch.cat([key, output], dim=1)
        #     output = self.conv1(output) + self.conv2(output)
        return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
    act_layer=nn.GELU,
    # act_layer=nn.SiLU,
     drop=0.6):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # self.sigmoid = nn.Sigmoid()

    # mask_ratio丢弃率
    def forward(self, query, key, value):
        B, N, C = query.shape
        query = query.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # q、k、v=[b,heads,n,c//heads]
        # attn = [B, heads, n, n]
        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # if use_sigmoid:
        #     # 这里输入的query2=query,key2=key,只不过用了单独的nn.Conv2d来学习他们的weight
        #     query2 = query2.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #     key2 = key2.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #     attn2 = torch.matmul(query2, key2.transpose(-2, -1)) * self.scale

        #     # ....我这里可以像convSp一样，搞个conv2d，然后sigmoid(conv2d(attn[B, heads, n, n]))后得到attn的权重z，然后再attn = attn * z ===<<<称之为自适应多头注意力机制---屎

        #     # torch.sum(attn2, dim=-1): attn2[B, heads, n, n] => attn2[B, heads, n]，求和之后，消除了最后一个维度，并且计算出了每一个dim2对整个dim3的关系?dim2为query，dim3为key。
        #     attn2 = torch.sum(attn2, dim=-1)

        #     # ---<<<这里我能否再计算一个attn3 = self.sigmoid(torch.sum(attn2, dim=-2))，即计算每一个key对query的关系，然后 attn = attn * attn2 * attn3
        #     # （这里我可以试试看
        #     # 计算各个query的权重
        #     attn2 = self.sigmoid(attn2)

        #     # attn2.unsqueeze(3)将attn2从[B, heads, n]=>[B, heads, n, 1]并broadcast与attn维度一致
        #     # attn=[B, heads, n, n]
        #     attn = attn * (attn2.unsqueeze(3))

        x = torch.matmul(attn, value).transpose(1, 2).reshape(B, N, C)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm,drop = 0.6,injection=False):
        super().__init__()

        self.channels = dim
        self.injection = injection

        self.encode_value = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.encode_query = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.encode_key = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

        # if self.injection:
        #     self.encode_query2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        #     self.encode_key2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,drop = drop)

        # self.q_embedding = nn.Parameter(torch.randn(1, 256, 32, 32))
        # self.k_embedding = nn.Parameter(torch.randn(1, 256, 32, 32))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, query_embed=None, key_embed=None):
        b, c, h, w = query.shape

        # query_embed = repeat(self.q_embedding, '() n c d -> b n c d', b = b) # 加embedding好像训练的一开始效果会很差，可能是parameter还没调整过来的原因?
        # key_embed = repeat(self.k_embedding, '() n c d -> b n c d', b = b)

        q_embed = self.with_pos_embed(query, query_embed)
        k_embed = self.with_pos_embed(key, key_embed)
        
        v = self.encode_value(key).view(b, self.channels, -1)
        v = v.permute(0, 2, 1)

        q = self.encode_query(q_embed).view(b, self.channels, -1) # q = [b,c,H*W]
        q = q.permute(0, 2, 1) # q = [b,H*W,c,]

        k = self.encode_key(k_embed).view(b, self.channels, -1)
        k = k.permute(0, 2, 1)
        #print(1)
        query = query.view(b, self.channels, -1).permute(0, 2, 1)

        # if self.injection:
        #     q2 = self.encode_query2(q_embed).view(b, self.channels, -1)
        #     q2 = q2.permute(0, 2, 1)

        #     k2 = self.encode_key2(k_embed).view(b, self.channels, -1)
        #     k2 = k2.permute(0, 2, 1)

        #     query = self.attn(query=q, key=k, value=v,query2 = q2, key2 = k2, use_sigmoid=True)
        # else:
        #     q2 = None
        #     k2 = None

            # query = query + self.attn(query=q, key=k, value=v, query2 = q2, key2 = k2, use_sigmoid=False)
        query = query + self.attn(query=q, key=k, value=v)
        #print(2)
        query = query + self.mlp(self.norm2(query))
        #print(3)
        query = query.permute(0, 2, 1).contiguous().view(b, self.channels, h, w)

        return query


class PositionEmbedding(nn.Module):
    def __init__(self, inp_res=32, num_channels=128):
        super(PositionEmbedding, self).__init__()
        self.row_embed = nn.Embedding(inp_res, num_channels)
        self.col_embed = nn.Embedding(inp_res, num_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.row_embed.weight)
        nn.init.zeros_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super(AttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.bn_layer = bn_layer

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.down_x = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                                kernel_size=1, stride=1, padding=0)
        self.down_y = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                                kernel_size=1, stride=1, padding=0)

        if self.bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0) # 0-initialization for residual block
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

    def forward(self, x, y):
        batch_size = x.size(0)

        g_y = self.g(y).view(batch_size, self.inter_channels, -1)
        g_y = g_y.permute(0, 2, 1)

        theta_x = self.down_x(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_y = self.down_y(y).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_y)
        f_div_C = F.softmax(f, dim=-1)

        y_2 = torch.matmul(f_div_C, g_y)
        y_2 = y_2.permute(0, 2, 1).contiguous()
        y_2 = y_2.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y_2 = self.W(y_2)
        z = W_y_2 + x

        return z