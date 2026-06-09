from transformers import AutoTokenizer

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1)+fibonacci(n-2)",
    "こんにちは世界",
    "नमस्ते दुनिया",
]

models = [
    "gpt2",
    "bert-base-uncased",
    "meta-llama/Llama-3.1-8B",
    "google/gemma-2-9b",
]

for model_name in models:
    print(f"\n=== {model_name} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)

        print(
            f"tokens={len(tokens):3d} | "
            f"{repr(text[:50])}"
        )
