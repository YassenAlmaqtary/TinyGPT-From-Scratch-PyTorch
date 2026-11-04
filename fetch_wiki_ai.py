"""
تحميل مقالات ويكيبيديا (إنجليزي) عن الذكاء الاصطناعي وحفظها في wiki_ai.txt.
نص حقيقي كتبه بشر = تنوع أكبر بكثير من النصوص المولّدة بالقوالب.

التشغيل:
    python fetch_wiki_ai.py
"""

import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

TITLES = [
    "Artificial intelligence",
    "History of artificial intelligence",
    "Artificial general intelligence",
    "Machine learning",
    "Deep learning",
    "Neural network (machine learning)",
    "Perceptron",
    "Multilayer perceptron",
    "Activation function",
    "Rectifier (neural networks)",
    "Softmax function",
    "Backpropagation",
    "Gradient descent",
    "Stochastic gradient descent",
    "Loss function",
    "Cross-entropy",
    "Overfitting",
    "Bias–variance tradeoff",
    "Regularization (mathematics)",
    "Dilution (neural networks)",
    "Early stopping",
    "Hyperparameter (machine learning)",
    "Batch normalization",
    "Residual neural network",
    "Supervised learning",
    "Unsupervised learning",
    "Self-supervised learning",
    "Reinforcement learning",
    "Deep reinforcement learning",
    "Transfer learning",
    "Fine-tuning (deep learning)",
    "Feature learning",
    "Convolutional neural network",
    "Recurrent neural network",
    "Long short-term memory",
    "Seq2seq",
    "Attention (machine learning)",
    "Transformer (deep learning architecture)",
    "Vision transformer",
    "Mixture of experts",
    "Language model",
    "Large language model",
    "Generative pre-trained transformer",
    "GPT-3",
    "GPT-4",
    "ChatGPT",
    "BERT (language model)",
    "Word embedding",
    "Word2vec",
    "Byte pair encoding",
    "Natural language processing",
    "Machine translation",
    "Speech recognition",
    "Computer vision",
    "Autoencoder",
    "Generative adversarial network",
    "Diffusion model",
    "Prompt engineering",
    "Retrieval-augmented generation",
    "Hallucination (artificial intelligence)",
    "AI alignment",
    "Ethics of artificial intelligence",
    "Explainable artificial intelligence",
    "Turing test",
    "Expert system",
    "Support vector machine",
    "Decision tree learning",
    "Random forest",
    "K-means clustering",
    "Statistical classification",
    "Data mining",
    "PyTorch",
    "TensorFlow",
    "Graphics processing unit",
    "CUDA",
    "AlphaGo",
    "Geoffrey Hinton",
    "Yann LeCun",
    "Yoshua Bengio",
    "Alan Turing",
]

STOP_SECTIONS = ("See also", "References", "External links", "Further reading", "Notes", "Bibliography", "Sources")

CHAR_MAP = {
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2026": "...", "\u00a0": " ",
}

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "TinyGPT-edu/0.1 (educational project)"}


def fetch_extract(title: str) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "titles": title,
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "")


def clean(raw: str) -> str:
    for src, dst in CHAR_MAP.items():
        raw = raw.replace(src, dst)
    # إزالة الحروف غير الإنجليزية حتى تبقى المفردات صغيرة
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")

    kept = []
    for line in raw.splitlines():
        header = re.match(r"^=+\s*(.*?)\s*=+$", line.strip())
        if header:
            if header.group(1) in STOP_SECTIONS:
                break
            continue
        # أسطر المعادلات الرياضية في النص الخام تكون مزاحة أو فيها displaystyle
        if line.startswith(" ") or "displaystyle" in line:
            continue
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) >= 40:
            kept.append(line)
    return "\n".join(kept)


def main():
    out_path = Path(__file__).resolve().parent / "wiki_ai.txt"
    articles = []
    for i, title in enumerate(TITLES, 1):
        try:
            text = clean(fetch_extract(title))
        except Exception as exc:
            print(f"[{i}/{len(TITLES)}] FAILED {title}: {exc}")
            continue
        if text:
            articles.append(text)
        print(f"[{i}/{len(TITLES)}] {title}: {len(text):,} chars")
        # ويكيبيديا تحدّ الطلبات؛ انتظر بين المقالات
        time.sleep(1.2)

    corpus = "\n\n".join(articles) + "\n"
    out_path.write_text(corpus, encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"articles={len(articles)} chars={len(corpus):,} vocab={len(set(corpus))}")


if __name__ == "__main__":
    main()
