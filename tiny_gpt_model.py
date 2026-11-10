import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, embedding_dim, head_dim, block_size, dropout=0.2):
        super().__init__()

        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        weights = self.dropout(F.softmax(scores, dim=-1))
        return weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size, dropout=0.2):
        super().__init__()

        assert embedding_dim % num_heads == 0
        head_dim = embedding_dim // num_heads

        self.heads = nn.ModuleList([
            Head(
                embedding_dim=embedding_dim,
                head_dim=head_dim,
                block_size=block_size,
                dropout=dropout,
            )
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        concatenated = torch.cat(outputs, dim=-1)
        return self.dropout(self.projection(concatenated))


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout=0.2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size, dropout=0.2):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForward(embedding_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=32,
        num_heads=4,
        block_size=8,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            Block(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                block_size=block_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx):
        _, T = idx.shape

        if T > self.block_size:
            raise ValueError("Sequence exceeds block_size")

        token_emb = self.token_embedding(idx)
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(positions)

        x = self.embedding_dropout(token_emb + pos_emb)
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.lm_head(x)


@torch.no_grad()
def generate(model, context, max_new_tokens, temperature=0.2):
    """
    temperature:
      - أقرب لـ 1.0 = تنوع أكثر
      - أقرب لـ 0.0 = نص أكثر حذرًا وتكرارًا
    """
    model.eval()
    for _ in range(max_new_tokens):
        input_tokens = context[:, -model.block_size:]
        logits = model(input_tokens)
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)
    return context
