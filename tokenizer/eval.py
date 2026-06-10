from transformers import AutoTokenizer
from tokenizers import Tokenizer

test_text = open("test.txt", encoding="utf-8").read()

baseline = AutoTokenizer.from_pretrained("gpt2")
custom = Tokenizer.from_file("custom_tokenizer.json")

baseline_tokens = len(
    baseline.encode(test_text, add_special_tokens=False)
)

custom_tokens = len(
    custom.encode(test_text).ids
)

print("baseline:", baseline_tokens)
print("custom  :", custom_tokens)

reduction = (
    (baseline_tokens - custom_tokens)
    / baseline_tokens
    * 100
)

print(f"token reduction: {reduction:.2f}%")
