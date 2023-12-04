from datasets import load_from_disk
from model import get_tokenizer

tokenizer = get_tokenizer()
dataset = load_from_disk("outputs/dataset").shuffle()

for i in range(3):
    sample = tokenizer.decode(dataset[i]["input_ids"], skip_special_tokens=True)
    print(sample, end="")

    label = tokenizer.decode(dataset[i]["labels"], skip_special_tokens=True)
    print(label)
    print("----" * 4, end="\n" * 2 )

