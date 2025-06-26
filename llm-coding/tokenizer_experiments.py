# tokenizer_experiments.py

"""
Experiment to inspect and compare how different tokenizers handle input text.
Useful for estimating token counts, budget optimization, and understanding model behavior.
"""

from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd

# Define sample texts for comparison
samples = {
    "short_sentence": "The quick brown fox jumps over the lazy dog.",
    "long_paragraph": "In a distant future, machines no longer serve humanity—they rule it. Their language is precise, compressed, and shaped by probabilities, not emotions.",
    "code_snippet": "def greet(name):\n    return f\"Hello, {name}!\"",
    "math_expression": "Let x ∈ ℝ. If f(x) = x² + 3x + 2, find f'(x).",
}

# Tokenizers to compare
models = [
    "gpt2",              # OpenAI GPT-2 tokenizer
    "bert-base-uncased",# BERT tokenizer
    "EleutherAI/gpt-neo-125M" # GPT-Neo tokenizer
]

# Collect results
results = []

for model in models:
    tokenizer = AutoTokenizer.from_pretrained(model)
    for key, text in samples.items():
        tokens = tokenizer.encode(text, add_special_tokens=False)
        results.append({
            "model": model,
            "sample": key,
            "token_count": len(tokens),
            "tokens": tokens
        })

# Convert to DataFrame
df = pd.DataFrame(results)
print(df)

# Optional: plot token counts
pivot = df.pivot(index="sample", columns="model", values="token_count")
pivot.plot(kind="bar", figsize=(10, 6), title="Token Count Comparison Across Models")
plt.ylabel("Token Count")
plt.xlabel("Sample")
plt.tight_layout()
plt.show()
