import torch
from torch import dot, nn
from torch.nn import functional as F
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request

# ============================================================
# 1. النص الذي سيتعلم منه النموذج
# ============================================================
# ندرّب على نص حقيقي عن الذكاء الاصطناعي (مقالات ويكيبيديا)،
# حتى تتوافق الـ prompts مثل "PyTorch" و"Neural networks" مع البيانات،
# وبدل النصوص المولّدة بالقوالب التي تشجّع على النسخ الحرفي.

data_path = Path(__file__).resolve().parent / "wiki_ai.txt"
if not data_path.exists():
    raise FileNotFoundError(
        f"Missing {data_path.name}. Run: python fetch_wiki_ai.py"
    )

text = data_path.read_text(encoding="utf-8")
print(f"Loaded {len(text):,} characters from {data_path.name}")

print("===== Training Data Preview =====")
print(text[:1000])
print("===============================")

# ============================================================
# 2. بناء Vocabulary
# ============================================================
# set(text) = الحروف الفريدة فقط
# sorted(...) = نرتبها حتى يكون الترتيب ثابتًا دائمًا

chars = sorted(list(set(text)))

vocab_size = len(chars)


print("Characters:")
print(chars)

print("\nVocabulary size:")
print(vocab_size)

# ============================================================
# 3. تحويل الحروف إلى أرقام
# ============================================================
# stoi: حرف → رقم
# itos: رقم → حرف

stoi = {ch: i for i, ch in enumerate(chars)}

print("\nString to Integer:")
print(stoi)

itos = {i: ch for i, ch in enumerate(chars)}

print("\nInteger to String:")
print(itos)

# ============================================================
# 4. Encoder / Decoder
# ============================================================

def encode(s):
    """
    تحويل نص إلى قائمة أرقام باستخدام stoi.
    إذا وُجد حرف غير معروف نوضح الحروف الناقصة.
    """
    missing = sorted({c for c in s if c not in stoi})
    if missing:
        raise ValueError(
            f"Characters {missing} not in vocabulary.\n"
            f"Add them to the training text, or change the sample."
        )
    return [stoi[c] for c in s]


def decode(ids):
    """تحويل قائمة أرقام إلى نص باستخدام itos."""
    return "".join(itos[i] for i in ids)


# الآن العينة صالحة لأن النص أعلاه يحتوي على 'A' و 'I' وباقي الحروف
#sample = "I love AI"

encoded = encode(text)
decoded = decode(encoded)

#print("\nOriginal:")
#print(sample)

print("\nEncoded (first 100 ids):")
print(encoded[:100])

print("\nDecoded sample (first 200 chars):")
print(decoded[:200])

data = torch.tensor(encoded, dtype=torch.long)

print("\nData shape:")
print(data.shape)

# نافذة أطول + دفعة أكبر تساعد النموذج يتعلم أنماطًا أغنى
block_size = 128
batch_size = 64
shape = [batch_size, block_size]

x = data[:block_size]
y = data[1:block_size + 1] # y is the next character

print("\nInput x:")
print(x)

print("\nTarget y:")
print(y)

print("\nX shape:")
print(x.shape)

print("\nY shape:")
print(y.shape)

# اختيار 4 مواقع عشوائية من النص
ix = torch.randint(high=len(data)-block_size, size=(batch_size,))

print("\nRandom positions:")
print(ix)

x_batch = torch.stack([data[i:i + block_size] for i in ix])
y_batch = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

print("\nX batch:")
print(x_batch)

print("\nY batch:")
print(y_batch)

print("\nX  batch shape:")
print(x_batch.shape)

print("\nY batch shape:")
print(y_batch.shape)

token_embedding =nn.Embedding(vocab_size, 32)

token_embeddings = token_embedding(x_batch)

# print("\nToken embeddings:")
# print(token_embeddings)

print("\nToken embeddings shape:")
print(token_embeddings.shape)

position_embedding = nn.Embedding(
    block_size,
    32
)

positions = torch.arange(block_size)

position_embeddings = position_embedding(positions)

# print("\nPosition embeddings:")
# print(position_embeddings)

print("\nPosition embeddings shape:")
print(position_embeddings.shape)

x = token_embeddings + position_embeddings

# print("\nFinal input to Transformer:")
# print(x)

print("\nFinal input shape x:")
print(x.shape)

# # ============================================================
# # Self-Attention: Q, K, V
# # ============================================================

# W_Q = nn.Linear(32, 32, bias=False)
# W_K = nn.Linear(32, 32, bias=False)
# W_V = nn.Linear(32, 32, bias=False)

# Q = W_Q(x)
# K = W_K(x)
# V = W_V(x)

# print("\nQ shape:")
# print(Q.shape)

# print("\nK shape:")
# print(K.shape)

# print("\nV shape:")
# print(V.shape)


# # .transpose(-2, -1) تعني تبديل البعدين الأخيرين (عادةً الطول والخصائص: sequence, feature)
# # -2 و -1 هما ترقيم للأبعاد من النهاية: -1 هو آخر بعد، -2 قبل الأخير (أي (batch, seq, feature) تصبح (batch, feature, seq))
# # لم نستخدم .T لأن .T في بايثون/بايتورتش تقلب جميع الأبعاد فقط إذا كان التنسور 2D
# # إذا كان التنسور أكثر من بعدين (مثل 3D: batch, seq, feature)، .T لا تتصرف كما نريد؛ لذلك نستخدم transpose(-2, -1)
# sscores = Q @ K.transpose(-2, -1)

# print("\nAttention scores shape:")
# print(sscores.shape)

# print("\nAttention scores for the first example:")
# print(sscores[0])

# # ============================================================
# # Causal Mask
# # ============================================================

# mask = torch.tril(
#     torch.ones(block_size, block_size)
# )

# print("\nCausal mask:")
# print(mask)

# sscores = sscores / torch.sqrt(torch.tensor(32))
# sscores = sscores.masked_fill(
#     mask == 0,
#     float('-inf')
# )

# print("\nMasked scores for the first example:")
# print(sscores[0])

# attention_weights = torch.softmax(
#     sscores,
#     dim=-1
# )


# print("\nAttention weights for the first example:")
# print(attention_weights[0])

# print("\nRow sums:")
# print(attention_weights[0].sum(dim=-1))





# # ============================================================
# # Weighted Sum of Values
# # ============================================================

# attention_output = attention_weights @ V

# print("\nAttention output shape:")
# print(attention_output.shape)

# print("\nAttention output for the first example:")
# print(attention_output[0])


# num_heads = 4
# head_dim = 32 // num_heads
# print("head_dim:", head_dim)
# Q_heads = Q.reshape(
#     batch_size,
#     block_size,
#     num_heads,
#     head_dim
# ).transpose(1, 2)

# K_heads = K.reshape(
#     batch_size,
#     block_size,
#     num_heads,
#     head_dim
# ).transpose(1, 2)

# V_heads = V.reshape(
#     batch_size,
#     block_size,
#     num_heads,
#     head_dim
# ).transpose(1, 2)


# print("\nQ_heads shape:")
# print(Q_heads.shape)

# print("\nK_heads shape:")
# print(K_heads.shape)

# print("\nV_heads shape:")
# print(V_heads.shape)

# scores_heads=Q_heads @  K_heads.transpose(-2,-1)

# print("\nAttention scores Head shape:")
# print(scores_heads.shape)

# scores_heads = scores_heads / torch.sqrt(torch.tensor(head_dim))

# print("\nAttention scores Head for the first example:")
# print(scores_heads[0])


# scores_heads = scores_heads.masked_fill(mask == 0, float('-inf'))
# print("\nAttention scores Head for the first example after masking:")
# print(scores_heads[0])

# attention_weights_heads = torch.softmax(scores_heads, dim=-1)

# print("\nAttention weights Head for the first example:")
# print(attention_weights_heads[0])

# attention_output_heads = attention_weights_heads @ V_heads

# print("\nAttention output Head shape:")
# print(attention_output_heads.shape)
# print("\nAttention output Head for the first example:")
# print(attention_output_heads[0])


# attention_output_heads = attention_output_heads.transpose(1, 2).reshape(batch_size, block_size, -1)

# print("\nAttention output Head reshaped for the first example:")
# print(attention_output_heads[0])

# print("\nAttention output Head shape:")
# print(attention_output_heads.shape)


# W_O = nn.Linear(
#     32,
#     32,
#     bias=False
# )

# output = W_O(attention_output_heads)

# print("\nFinal Multi-Head Attention output shape:")
# print(output.shape)

# print("\nFinal Multi-Head Attention output:")
# print(output)

# residual_output = x + output

# print("\nResidual output shape:")
# print(residual_output.shape)

# print("\nResidual output:")
# print(residual_output)


# layer_norm = nn.LayerNorm(32)
# normalized_output = layer_norm(residual_output)

# print("\nNormalized output shape:")
# print(normalized_output.shape)

# print("\nNormalized output:")
# print(normalized_output)

# ffn_linear1 = nn.Linear(
#     32,
#     128
# )

# ffn_hidden = ffn_linear1(normalized_output)

# print("\nFFN hidden shape:")
# print(ffn_hidden.shape)

# ffn_activation = torch.nn.functional.gelu(ffn_hidden)

# print("\nFFN activation shape:")
# print(ffn_activation.shape)

# ffn_linear2 = nn.Linear(
#     128,
#     32
# )

# ffn_output = ffn_linear2(ffn_activation)

# print("\nFFN output shape:")
# print(ffn_output.shape)

# print("\nFFN output:")
# print(ffn_output)


# block_output = normalized_output + ffn_output

# print("\nBlock output shape:")
# print(block_output.shape)

# final_norm=nn.LayerNorm(32)
# block_output = final_norm(block_output)

# print("\nFinal Transformer Block output:")
# print(block_output[0,0:])

# lm_head = nn.Linear(
#     32,
#     vocab_size
# )

# logits = lm_head(block_output)

# print("\nLogits shape:", logits.shape)

# probabilities = torch.softmax(logits, dim=-1)

# print("\nProbabilities shape:", probabilities.shape)

# print("\nProbabilities for the first example, first position:")
# print(probabilities[0, 0])

# print("\nSum of probabilities:")
# print(probabilities[0, 0].sum())

# loss_fn = nn.CrossEntropyLoss()

# loss = loss_fn(
#     logits.reshape(-1, vocab_size),
#     y_batch.reshape(-1)
# )

# print("\nLoss:") 
# print(loss.item())

# params = (
#     list(token_embedding.parameters())
#     + list(position_embedding.parameters())
#     + list(W_Q.parameters())
#     + list(W_K.parameters())
#     + list(W_V.parameters())
#     + list(W_O.parameters())
#     + list(layer_norm.parameters())
#     + list(ffn_linear1.parameters())
#     + list(ffn_linear2.parameters())
#     + list(final_norm.parameters())
#     + list(lm_head.parameters())
# )

# optimizer = torch.optim.AdamW(
#     params,
#     lr=0.001
# )

# print("\nTotal parameter tensors:")
# print(len(params))

# optimizer.zero_grad()

# loss.backward()

# print("\nGradient of lm_head:")
# print(lm_head.weight.grad.shape)

# print("\nGradient norm:")
# print(lm_head.weight.grad.norm().item())

# old_weight = lm_head.weight.detach().clone()

# optimizer.step()

# weight_change = (
#     lm_head.weight.detach() - old_weight
# ).norm().item()

# print("\nWeight change:")
# print(weight_change)



class Head(nn.Module):
    def __init__(self, embedding_dim, head_dim, block_size, dropout=0.2):
        super().__init__()

        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)

        # Causal mask: token i cannot see future tokens.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape

        Q = self.query(x)  # [B, T, head_dim]
        K = self.key(x)    # [B, T, head_dim]
        V = self.value(x)  # [B, T, head_dim]

        scores = Q @ K.transpose(-2, -1)
        scores = scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        output = weights @ V

        return output



class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size, dropout=0.2):
        super().__init__()

        assert embedding_dim % num_heads == 0 #معيارية: يجب أن يكون embedding_dim قابلًا للقسمة على num_heads

        head_dim = embedding_dim // num_heads

        self.heads = nn.ModuleList([
            Head(
                embedding_dim=embedding_dim,
                head_dim=head_dim,
                block_size=block_size,
                dropout=dropout,
            )
            for _ in range(num_heads) # إنشاء عدد من الرؤوس
        ])

        self.projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, embedding_dim]

        # تشغيل جميع الرؤوس ودمج نتائجها
        outputs = [head(x) for head in self.heads]

        # كل رأس ينتج [B, T, head_dim]
        # بعد الدمج يصبح [B, T, embedding_dim]
        concatenated = torch.cat(outputs, dim=-1) # دمج نتائج جميع الرؤوس

        # مزج المعلومات القادمة من جميع الرؤوس 
        #[B, T, embedding_dim] → [B, T, embedding_dim]
        output = self.dropout(self.projection(concatenated))

        return output


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
        # المرحلة الأولى: الانتباه مع Residual Connection
        x = x + self.attention(self.ln1(x))

        # المرحلة الثانية: FeedForward مع Residual Connection
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

        # تحويل رقم كل حرف إلى متجه
        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim
        )

        # إضافة معلومات الموقع لكل حرف
        self.position_embedding = nn.Embedding(
            block_size, embedding_dim
        )

        # تطبيق Dropout على التمثيلات المدمجة
        self.embedding_dropout = nn.Dropout(dropout)

        # تكديس كتل Transformer
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

        # تحويل التمثيل إلى درجات الأحرف المحتملة
        self.lm_head = nn.Linear(
            embedding_dim, vocab_size
        )

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError("Sequence exceeds block_size")

        token_emb = self.token_embedding(idx)  # [B,T,C]

        positions = torch.arange(
            T, device=idx.device
        )
        pos_emb = self.position_embedding(positions)  # [T,C]

        x = self.embedding_dropout(token_emb + pos_emb)
        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.lm_head(x)  # [B,T,vocab_size]

        return logits




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
        # عيّنة عشوائية (وليس argmax) حتى لا يعلق النموذج في تكرار
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)
    return context



# model = TinyGPT(
#     vocab_size=vocab_size,
#     embedding_dim=32,
#     num_heads=4,
#     block_size=8,
#     num_layers=2
# )

# اختر GPU إذا كان PyTorch يستطيع استخدامه
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# model = model.to(device)

# logits = model(x_batch.to(device))

# # دالة قياس الخطأ والمُحسّن
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=0.001
# )
# 1) التقسيم أولًا

if __name__ == "__main__":
    split_index = int(0.9 * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]
    print(len(train_data), len(val_data))

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)




    @torch.no_grad()
    def estimate_loss(model, train_data, val_data, eval_iters=100):
        model.eval()
        results = {}

        for split_name, split_data in [("train", train_data), ("val", val_data)]:
            losses = []
            for _ in range(eval_iters):
                x, y = get_batch(split_data)
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
                losses.append(loss.item())
            results[split_name] = sum(losses) / len(losses)

        model.train()
        return results


    # 2) نموذج جديد تمامًا (سعة أكبر قليلًا مع بيانات ويكيبيديا)
    model = TinyGPT(
        vocab_size=vocab_size,
        embedding_dim=128,
        num_heads=4,
        block_size=block_size,
        num_layers=6,
        dropout=0.2,
    ).to(device)


    total_params = sum(
        p.numel() for p in model.parameters()
    )


    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTotal parameters: {total_params:,}")
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"\nModel size: {trainable_params / 1_000_000:.3f} million")

    print("\nTotal characters:", len(data))
    print("\nTraining characters:", len(train_data))
    print("\nValidation characters:", len(val_data))

    print("\nTraining preview:")
    print(decode(train_data[:300].tolist()))

    print("\nValidation preview:")
    print(decode(val_data[:300].tolist()))

    

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # فحص سلامة القياس: يجب أن يكون الرقمان قرب ln(vocab_size)
    print(estimate_loss(model, train_data, val_data))

    # بيانات ويكيبيديا أكبر تحتاج تدريبًا أطول
    num_steps = 20000
    eval_interval = 500
    patience = 8

    best_val = float("inf")
    best_step = 0
    best_state = None
    checks_without_improvement = 0

    steps_history = []
    train_losses = []
    val_losses = []


    model.train()

    for step in range(num_steps):
        x_batch, y_batch = get_batch(train_data)

        logits = model(x_batch)
        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            y_batch.reshape(-1)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)

            current_step = step + 1

            steps_history.append(current_step)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])

            print(
                f"Step {step + 1}: "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}"
            )

            if losses["val"] < best_val:
                best_val = losses["val"]
                best_step = step + 1
                best_state = copy.deepcopy(model.state_dict())
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1

            # estimate_loss يضع النموذج في وضع التقييم

            if checks_without_improvement >= patience:
                print("Early stopping: validation loss stopped improving.")
                break

    plt.figure(figsize=(10, 5))

    plt.plot(
        steps_history,
        train_losses,
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        steps_history,
        val_losses,
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("TinyGPT Training and Validation Loss")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("tiny_gpt_loss.png", dpi=150)
    plt.show()

    # احفظ أفضل أوزان بعد التدريب على بيانات الذكاء الاصطناعي
    Path("models").mkdir(exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), "models/tiny_gpt_best.pth")
    print(f"Best step: {best_step}")
    print(f"Best validation loss: {best_val:.4f}")
    print("Best model saved to models/tiny_gpt_best.pth")


    # 1. تحميل أفضل نسخة من النموذج
    checkpoint = torch.load(
        "models/tiny_gpt_best.pth",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    # 2. بدايات من نفس مجال بيانات التدريب (الذكاء الاصطناعي)
    prompts = [
        "PyTorch ",
        "A transformer ",
        "Deep learning ",
        "Neural networks ",
        "Language models ",
    ]

    print("\n===== TinyGPT Generation Test =====\n")

    for prompt in prompts:
        encoded = encode(prompt)
        context = torch.tensor(
            [encoded],
            dtype=torch.long,
            device=device
        )

        output = generate(
            model,
            context,
            max_new_tokens=200,
            temperature=0.8,
        )

        generated_text = decode(output[0].tolist())

        print(f"Prompt: {prompt!r}")
        print(f"Generated:\n{generated_text}")
        print("-" * 60)


    # lm_head = nn.Linear(32, vocab_size)

    # # ============================================================
    # # 6. حلقة التدريب (Training Loop)
    # # ============================================================
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(
    #     list(token_embedding.parameters()) + 
    #     list(position_embedding.parameters()) + 
    #     list(W_Q.parameters()) + 
    #     list(W_K.parameters()) + 
    #     list(W_V.parameters()) + 
    #     list(lm_head.parameters()), 
    #     lr=1e-3
    # )

    # print("Starting Training Loop...")
    # for step in range(300): # زيادة عدد الدورات قليلاً ليتعلم بشكل أفضل
    #     token_embeddings = token_embedding(x_batch)
    #     x = token_embeddings + position_embedding(torch.arange(x_batch.shape[1]))

    #     Q = W_Q(x)
    #     K = W_K(x)
    #     V = W_V(x)

    #     sscores = Q @ K.transpose(-2, -1)
    #     sscores = sscores / torch.sqrt(torch.tensor(32, dtype=torch.float32))
    #     current_block_size = x_batch.shape[1]
    #     current_mask = torch.tril(torch.ones(current_block_size, current_block_size))
    #     sscores = sscores.masked_fill(current_mask == 0, float('-inf'))

    #     attention_weights = torch.softmax(sscores, dim=-1)
    #     attention_output = attention_weights @ V

    #     logits = lm_head(attention_output)

    #     loss = loss_fn(logits.view(-1, vocab_size), y_batch.view(-1))

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    # print(f"Training finished! Final Loss: {loss.item():.4f}\n")

    # # ============================================================
    # # 7. دالة التوليد (Text Generation - Next Token)
    # # ============================================================
    # def generate(model_inputs, max_new_tokens):
    #     # model_inputs: الحضانة أو النص البدئي المرمز (Tensor)
    #     for _ in range(max_new_tokens):
    #         # نأخذ آخر block_size رموز فقط ليتناسب مع حجم الـ Positional Embedding
    #         idx_cond = model_inputs[:, -block_size:]
    #         curr_len = idx_cond.shape[1]

    #         tok_emb = token_embedding(idx_cond)
    #         pos_emb = position_embedding(torch.arange(curr_len))
    #         x = tok_emb + pos_emb

    #         Q = W_Q(x)
    #         K = W_K(x)
    #         V = W_V(x)

    #         curr_mask = torch.tril(torch.ones(curr_len, curr_len))
    #         sscores = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(32, dtype=torch.float32))
    #         sscores = sscores.masked_fill(curr_mask == 0, float('-inf'))

    #         att_weights = torch.softmax(sscores, dim=-1)
    #         att_output = att_weights @ V

    #         logits = lm_head(att_output)

    #         # نركز على الخطوة الزمنية الأخيرة فقط (-1) للحصول على توقع الحرف التالي
    #         logits = logits[:, -1, :] # الشكل يصبح: (Batch, Vocab_Size)
    #         probs = torch.softmax(logits, dim=-1)

    #         # اختيار الرمز التالي بناءً على الاحتمالات
    #         next_token = torch.multinomial(probs, num_samples=1) # (Batch, 1)

    #         # دمج الرمز الجديد مع التسلسل الحالي
    #         model_inputs = torch.cat((model_inputs, next_token), dim=1)

    #     return model_inputs

    # # اختبار التوليد بنص بداعي (مثلاً أول حرفين "I ")
    # start_text = "I "
    # context = torch.tensor([encode(start_text)], dtype=torch.long)

    # generated_ids = generate(context, max_new_tokens=50)
    # generated_text = decode(generated_ids[0].tolist())

    # print("Generated Text:")
    # print("----------------------------------------")
    # print(generated_text)
    # print("----------------------------------------")
