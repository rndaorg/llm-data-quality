from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=32000,
    special_tokens=["<pad>", "<unk>"]
)

tokenizer.train(
    files=["corpus.txt"],
    trainer=trainer
)

tokenizer.save("custom_tokenizer.json")
