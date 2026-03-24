import math
import torch
import torch.nn as nn

from .vit import Add, MatMul, MatMulIsqrtD


class BertGetFirstToken(nn.Module):
    def __init__(self):
        super(BertGetFirstToken, self).__init__()

    def forward(self, x):
        return x[:, :1]


class BertEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        type_vocab_size: int,
    ):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.embd_add1 = Add()
        self.embd_add2 = Add()

    def forward(self, input_ids, token_type_ids=None):
        batch_size, seq_len = input_ids.shape
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        word = self.word_embeddings(input_ids)
        pos = self.position_embeddings(position_ids)
        tok = self.token_type_embeddings(token_type_ids)
        out = self.embd_add1(word, pos)
        out = self.embd_add2(out, tok)
        return self.norm(out)


class BertMaskedSoftmax(nn.Module):
    def __init__(self, dim: int = -1):
        super(BertMaskedSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x, attention_bias=None):
        if attention_bias is not None:
            x = x + attention_bias
        return torch.softmax(x, dim=self.dim)


class BertAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super(BertAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)

        self.softmax = BertMaskedSoftmax(dim=-1)
        self.matmul1 = MatMulIsqrtD(self.head_dim)
        self.matmul2 = MatMul()

    def split_heads(self, x):
        new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, attention_bias=None):
        q = self.split_heads(self.linear_q(x))
        k = self.split_heads(self.linear_k(x))
        v = self.split_heads(self.linear_v(x))

        attn_score = self.matmul1(q, k.transpose(-2, -1))
        attn_prob = self.softmax(attn_score, attention_bias)

        context = self.matmul2(attn_prob, v)

        context = context.permute(0, 2, 1, 3)
        context = context.reshape(context.shape[0], context.shape[1], -1)
        return self.linear_out(context)


class BertFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super(BertFFN, self).__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class BertLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = BertFFN(hidden_size, intermediate_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.add1 = Add()
        self.add2 = Add()

    def forward(self, x, attention_bias=None):
        attn_out = self.attention(x, attention_bias)
        out1 = self.norm1(self.add1(x, attn_out))
        ffn_out = self.ffn(out1)
        out2 = self.norm2(self.add2(out1, ffn_out))
        return out2


class BertEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, num_layers: int):
        super(BertEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [BertLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)]
        )

    def forward(self, x, attention_bias=None):
        for layer in self.layers:
            x = layer(x, attention_bias)
        return x


class BertPooler(nn.Module):
    def __init__(self, hidden_size: int):
        super(BertPooler, self).__init__()
        self.get_first_token = BertGetFirstToken()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_token = self.get_first_token(x).squeeze(1)
        return self.activation(self.dense(cls_token))


class BertModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_layers: int,
        max_position_embeddings: int,
        type_vocab_size: int,
    ):
        super(BertModel, self).__init__()
        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
        )
        self.encoder = BertEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
        )
        self.pooler = BertPooler(hidden_size)

    @staticmethod
    def build_attention_bias(attention_mask: torch.Tensor, dtype: torch.dtype):
        if attention_mask is None:
            return None
        valid_mask = attention_mask > 0
        bias = torch.zeros_like(attention_mask, dtype=dtype)
        neg_inf = torch.tensor(float("-inf"), device=attention_mask.device, dtype=dtype)
        bias = torch.where(valid_mask, bias, neg_inf)
        return bias[:, None, None, :]

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        hidden_states = self.embeddings(input_ids, token_type_ids)
        attention_bias = self.build_attention_bias(attention_mask, hidden_states.dtype)
        sequence_output = self.encoder(hidden_states, attention_bias)
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class BertForSequenceClassification(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_layers: int,
        max_position_embeddings: int,
        type_vocab_size: int,
        num_labels: int,
    ):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
        )
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        _, pooled_output = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(pooled_output)
        return logits
