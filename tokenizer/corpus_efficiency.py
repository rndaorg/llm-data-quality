from transformers import AutoTokenizer

models = [
    "gpt2",
    "meta-llama/Llama-3.1-8B",
    "google/gemma-2-9b",
]

with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read()

n_chars = len(corpus)

for model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ids = tokenizer.encode(
        corpus,
        add_special_tokens=False
    )

    n_tokens = len(ids)

    print(f"\n{model_name}")
    print(f"tokens      : {n_tokens:,}")
    print(f"chars/token : {n_chars / n_tokens:.3f}")
