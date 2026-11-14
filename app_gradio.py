"""واجهة Gradio لتوليد النص باستخدام TinyGPT المدرّب محليًا."""

import os
from pathlib import Path

# منع مرور طلبات localhost عبر Proxy النظام
for key in ("NO_PROXY", "no_proxy"):
    current = os.environ.get(key, "")
    extras = "127.0.0.1,localhost,::1"
    os.environ[key] = f"{current},{extras}" if current else extras

import gradio as gr
import torch

from tiny_gpt_model import TinyGPT, generate

PROJECT_DIR = Path(__file__).resolve().parent
CORPUS_PATH = PROJECT_DIR / "wiki_ai.txt"
if not CORPUS_PATH.exists():
    raise FileNotFoundError(
        f"Missing training corpus: {CORPUS_PATH}\n"
        "Run: python fetch_wiki_ai.py"
    )

text = CORPUS_PATH.read_text(encoding="utf-8")
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {char: index for index, char in enumerate(chars)}
itos = {index: char for char, index in stoi.items()}

block_size = 128
EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 6
DROPOUT = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = PROJECT_DIR / "models" / "tiny_gpt_best.pth"
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Missing checkpoint: {CHECKPOINT_PATH}\n"
        "Train first with: python tiny_gpt.py"
    )

model = TinyGPT(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    block_size=block_size,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(device)

state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"Loaded TinyGPT on {device} | vocab={vocab_size} | block_size={block_size}")


def encode(value: str) -> list[int]:
    missing = sorted(set(value) - stoi.keys())
    if missing:
        raise ValueError(
            f"Characters {missing} not in vocabulary. "
            "Use characters present in the training corpus."
        )
    return [stoi[char] for char in value]


def decode(ids: list[int]) -> str:
    return "".join(itos[index] for index in ids)


def generate_text(prompt: str, max_new_tokens: int, temperature: float) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return "اكتب بداية نص أولًا (مثل: PyTorch )"

    temperature = max(float(temperature), 0.05)
    max_new_tokens = int(max_new_tokens)

    try:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    except ValueError as exc:
        return str(exc)

    output = generate(
        model,
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return decode(output[0].tolist())


demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="البداية (Prompt)", value="PyTorch ", lines=2),
        gr.Slider(20, 400, value=150, step=10, label="عدد الأحرف الجديدة"),
        gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature"),
    ],
    outputs=gr.Textbox(label="النص المولَّد", lines=12),
    title="TinyGPT Demo",
    description=(
        "واجهة تستدعي TinyGPT المدرّب محليًا. "
        "مدرَّب على مقالات ويكيبيديا عن الذكاء الاصطناعي (Character-level)."
    ),
    examples=[
        ["PyTorch ", 200, 0.8],
        ["A transformer ", 200, 0.8],
        ["Deep learning ", 200, 0.7],
        ["Neural networks ", 200, 0.8],
        ["Language models ", 200, 0.9],
    ],
)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )
