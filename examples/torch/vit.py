import allo
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ViTModel(nn.Module):
    def __init__(self, n_embd, n_head, n_layers, n_chn, ptc_size, img_size):
        super(ViTModel, self).__init__()
        self.embeddings = ViTEmbedding(n_embd, n_chn=n_chn, ptc_size=ptc_size, img_size=img_size)
        self.vit_blocks = nn.ModuleList(
            [ViTBlock(n_embd, n_head, n_embd * 4) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.pooler = ViTPooler(n_embd, self.embeddings.n_ptc)

    def forward(self, x):
        x = self.embeddings(x)
        for block in self.vit_blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.pooler(x)
        return x
    

class ViTEmbedding(nn.Module):
    def __init__(self, n_embd, n_chn=3, ptc_size=(16,16), img_size=(224,224)):
        super(ViTEmbedding, self).__init__()
        self.n_ptc = (img_size[0] // ptc_size[0]) * (img_size[1] // ptc_size[1])
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd))
        # self.mask_t = nn.Parameter(torch.randn(1, 1, n_embd))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.n_ptc + 1, n_embd))
        self.proj = nn.Conv2d(n_chn, n_embd, kernel_size=ptc_size, stride=ptc_size)
        self.n_embd = n_embd
        self.expand1 = ViTTokenExpand(self.cls_token.shape)
        self.expand2 = ViTTokenExpand(self.position_embeddings.shape)

    def forward(self, x):
        embeddings = self.proj(x).view(x.shape[0], self.n_embd, -1).transpose(1, 2)
        # cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        cls_tokens = self.expand1(self.cls_token, x)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings += self.expand2(self.position_embeddings, x)
        return embeddings

class ViTTokenExpand(nn.Module):
    def __init__(self, token_shape):
        super(ViTTokenExpand, self).__init__()
        self.token_shape = token_shape # not include batch_size

    def forward(self, token, x):
        return token.expand(x.shape[0], -1, -1)

class ViTBlock(nn.Module):
    def __init__(self, n_embd, num_heads, ffn_hidden_dim):
        super(ViTBlock, self).__init__()
        self.attention = MultiHeadAttention(n_embd, num_heads)
        self.norm1 = nn.LayerNorm(n_embd)
        self.ffn = FFN(n_embd, ffn_hidden_dim, n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.norm1(x)
        attn_output = self.attention(x)

        out1 = x + attn_output
        out2 = self.norm2(out1)

        ffn_output = self.ffn(out2, out1)
        return ffn_output

class ViTPooler(nn.Module):
    def __init__(self, hidden_size, seq_len):
        super(ViTPooler, self).__init__()
        self.getFirstToken = ViTGetFirstToken(seq_len, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        first_token_tensor = self.getFirstToken(x)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ViTGetFirstToken(nn.Module):
    def __init__(self, seq_len, hidden_size):
        super(ViTGetFirstToken, self).__init__()
        self.shape = (1, seq_len + 1, hidden_size)

    def forward(self, x):
        return x[:, 0]

class FFN(nn.Module):
    def __init__(self, n_embd, hidden_dim, output_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(n_embd, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x, o):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x + o
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads

        self.linear_q = nn.Linear(n_embd, n_embd)
        self.linear_k = nn.Linear(n_embd, n_embd)
        self.linear_v = nn.Linear(n_embd, n_embd)

        self.linear_out = nn.Linear(n_embd, n_embd)

    def split_heads(self, x):
        # x: (batch_size, seq_len, hidden_size)
        new_shape = x.shape[:-1] + (self.num_heads, -1)
        x = x.view(new_shape)
        # output: (bs, head, seq, hs // head)
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product(self, q, k, v, x):
        # (bs, head, seq, hs // head)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        # (bs, head, seq, seq)
        attn_probs = F.softmax(attn_score, dim=-1)
        # (bs, head, seq, hs // head)
        attn = torch.matmul(attn_probs, v)
        return attn

    def forward(self, x):
        # qkv layers
        q = self.split_heads(self.linear_q(x))
        k = self.split_heads(self.linear_k(x))
        v = self.split_heads(self.linear_v(x))
        # core attention
        output = self.scaled_dot_product(q, k, v, x)
        # output: (bs, seq, head, hs // head)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(output.shape[0], output.shape[1], -1)
        output = self.linear_out(output)
        return output

n_embd = 768
n_head = 12
n_layers = 2 # 12
n_channels = 3
batch_size = 2
patch_size = (16, 16)
img_size = (224, 224)

# Correct!
# example_inputs = [torch.rand(batch_size, n_channels, img_size[0], img_size[1])]
# module = ViTEmbedding(n_embd, n_channels, patch_size, img_size).eval()

# Correct!
# example_inputs = [torch.rand(batch_size, n_channels, img_size[0], img_size[1])]
# class ConvTest(nn.Module):
#     def __init__(self, n_channels, n_embd, patch_size):
#         super(ConvTest, self).__init__()
#         self.conv2d = nn.Conv2d(n_channels, n_embd, kernel_size=patch_size, stride=patch_size, bias=False)
    
#     def forward(self, x):
#         return self.conv2d(x)
# module = ConvTest(n_channels, n_embd, patch_size).eval()

# int64
# module.conv2d.weight.requires_grad = False
# module.conv2d.bias.requires_grad = False
# module.conv2d.weight.data = torch.randint(-2, 2, (n_embd, n_channels, patch_size[0], patch_size[1])).type(torch.int8).detach()
# module.conv2d.bias.data = torch.randint(-100, 100, (n_embd,)).type(torch.int8).detach()
# example_inputs = [torch.randint(-2, 2, (batch_size, n_channels, img_size[0], img_size[1])).type(torch.int8)]
# module.conv2d.weight.data = torch.ones((n_embd, n_channels, patch_size[0], patch_size[1])).detach()
# module.conv2d.bias.data = torch.ones((n_embd,), dtype=torch.int8).detach() * 100
# example_inputs = [torch.ones((batch_size, n_channels, img_size[0], img_size[1]), dtype=torch.int8)]

# dsl conv2d
# import allo.dsl as dsl
# np_res = dsl.conv2d(example_inputs[0].detach().numpy(), module.conv2d.weight.data.detach().numpy(), module.conv2d.stride, module.conv2d.bias.data.detach().numpy())

# Correct!
# seq_len = int(torch.prod(torch.Tensor(img_size) // torch.Tensor(patch_size)).item())
# module = ViTBlock(n_embd, n_head, n_embd * 4).eval()
# example_inputs = [torch.rand(batch_size, seq_len + 1, n_embd)]

# Fault! 0.02? no Correct! just because the bias not added in conv2d originally
# Ok Correct, the reason is that the bias not added in the llvm process
example_inputs = [torch.rand(batch_size, n_channels, img_size[0], img_size[1])]
module = ViTModel(n_embd, n_head, n_layers, n_channels, patch_size, img_size).eval()

golden = module(*example_inputs)
llvm_mod = allo.frontend.from_pytorch(
    module,
    example_inputs=example_inputs,
    leaf_modules=[ViTGetFirstToken, ViTTokenExpand],
    verbose=False,
)
exit(0)

golden = module(*example_inputs)
np_inputs = [x.detach().numpy() for x in example_inputs]
res = llvm_mod(*np_inputs)
# np.testing.assert_allclose(res, np_res, atol=1e-3)
# np.testing.assert_allclose(np_res, golden.detach().numpy(), atol=1e-3)
np.testing.assert_allclose(res, golden.detach().numpy(), atol=1e-3)