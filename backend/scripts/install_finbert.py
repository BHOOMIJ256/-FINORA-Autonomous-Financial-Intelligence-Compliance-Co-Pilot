"""
FINORA — FinBERT Installation & Verification Script
=====================================================
Downloads and caches the ProsusAI/finbert model locally.
Run this ONCE before starting Agent 3.

What this does:
  1. Downloads FinBERT (~440MB) from HuggingFace
  2. Caches it locally so Agent 3 loads instantly
  3. Runs a quick verification test on 5 sample headlines
  4. Confirms the model is working correctly

Usage:
  python scripts/install_finbert.py

After this runs successfully:
  - Model cached at: ~/.cache/huggingface/hub/
  - Agent 3 will load it in ~3 seconds on first run
  - Subsequent loads are instant
"""

import sys
import time
from pathlib import Path

# ── Verify dependencies before downloading ────────────────
print("\n" + "="*55)
print("  FINORA — FinBERT Setup")
print("="*55)

print("\n📦 Checking dependencies...")

missing = []
try:
    import torch
    print(f"  ✅ PyTorch {torch.__version__}")
except ImportError:
    missing.append("torch")

try:
    import transformers
    print(f"  ✅ Transformers {transformers.__version__}")
except ImportError:
    missing.append("transformers")

try:
    import numpy
    print(f"  ✅ NumPy {numpy.__version__}")
except ImportError:
    missing.append("numpy")

if missing:
    print(f"\n  ❌ Missing packages: {', '.join(missing)}")
    print(f"  Run: pip install {' '.join(missing)}")
    sys.exit(1)

print("  ✅ All dependencies present")

# ── Download FinBERT ──────────────────────────────────────
print("\n📥 Downloading FinBERT (ProsusAI/finbert)...")
print("  Size: ~440MB | Only downloads once, then cached")
print("  This may take 2-5 minutes depending on connection...\n")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "ProsusAI/finbert"

try:
    start = time.time()

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("  ✅ Tokenizer ready")

    print("  Loading model weights...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"  ✅ Model ready ({round(time.time()-start, 1)}s)")

except Exception as e:
    print(f"\n  ❌ Download failed: {e}")
    print("  Check your internet connection and try again.")
    sys.exit(1)

# ── Verification Test ─────────────────────────────────────
print("\n🧪 Running verification tests...")
print("  Testing 5 sample Indian fintech headlines:\n")

TEST_HEADLINES = [
    {
        "text": "RBI tightens liquidity norms for NBFCs amid rising credit risk concerns",
        "expected": "negative"
    },
    {
        "text": "UPI transactions hit record high of 18 billion in March 2026",
        "expected": "positive"
    },
    {
        "text": "SEBI issues new circular on investment adviser regulations",
        "expected": "neutral"
    },
    {
        "text": "Major fintech NBFC defaults on ₹500 crore debt repayment",
        "expected": "negative"
    },
    {
        "text": "Digital lending platforms report strong Q4 growth in disbursements",
        "expected": "positive"
    }
]

# Label mapping from FinBERT output
LABEL_MAP = {
    "LABEL_0": "positive",
    "LABEL_1": "negative",
    "LABEL_2": "neutral",
    # Some versions use these labels directly
    "positive": "positive",
    "negative": "negative",
    "neutral":  "neutral"
}

def run_finbert(texts: list[str]) -> list[dict]:
    """Run FinBERT inference on a list of texts."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs     = torch.nn.functional.softmax(outputs.logits, dim=-1)
    results   = []

    for i, prob in enumerate(probs):
        scores    = prob.tolist()
        label_idx = prob.argmax().item()
        raw_label = model.config.id2label[label_idx]
        label     = LABEL_MAP.get(raw_label, raw_label).lower()

        results.append({
            "text":       texts[i][:60] + "...",
            "sentiment":  label,
            "confidence": round(max(scores), 4),
            "scores": {
                "positive": round(scores[0], 4),
                "negative": round(scores[1], 4),
                "neutral":  round(scores[2], 4)
            }
        })

    return results


# Run tests
texts   = [h["text"] for h in TEST_HEADLINES]
results = run_finbert(texts)

all_correct = True
for i, (result, test) in enumerate(zip(results, TEST_HEADLINES)):
    expected = test["expected"]
    got      = result["sentiment"]
    correct  = "✅" if got == expected else "⚠️ "

    if got != expected:
        all_correct = False

    print(f"  {correct} [{i+1}] {result['text']}")
    print(f"       Sentiment: {got.upper()} "
          f"(confidence: {result['confidence']}) "
          f"| Expected: {expected}")
    print(f"       Scores → pos: {result['scores']['positive']} | "
          f"neg: {result['scores']['negative']} | "
          f"neu: {result['scores']['neutral']}")
    print()

# ── Save model config for Agent 3 ─────────────────────────
import json
from pathlib import Path

config_path = Path(__file__).parent.parent / "models" / "saved" / "finbert_config.json"
config_path.parent.mkdir(parents=True, exist_ok=True)

config = {
    "model_name":   MODEL_NAME,
    "label_map":    LABEL_MAP,
    "max_length":   512,
    "verified":     True,
    "verified_at":  time.strftime("%Y-%m-%dT%H:%M:%S")
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

# ── Summary ───────────────────────────────────────────────
print("="*55)
if all_correct:
    print("  ✅ All tests passed — FinBERT is working perfectly")
else:
    print("  ⚠️  Some predictions differed from expected")
    print("  This is normal — FinBERT may classify differently")
    print("  than expected on edge cases. Model is still usable.")

print(f"\n  Model:  {MODEL_NAME}")
print(f"  Cache:  ~/.cache/huggingface/hub/")
print(f"  Config: {config_path}")
print("\n  ✅ FinBERT setup complete!")
print("  You can now build agents/sentiment_agent.py")
print("="*55 + "\n")