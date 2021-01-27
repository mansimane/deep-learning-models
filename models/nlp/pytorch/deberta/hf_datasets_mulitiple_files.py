from torch.utils.data import Dataset, IterableDataset, DataLoader
from itertools import cycle, islice
from transformers import DebertaTokenizer, DebertaModel
import torch
from transformers import DebertaConfig
from transformers import (
    DebertaConfig,
    DebertaTokenizer,
    DebertaForMaskedLM,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import os
from itertools import chain

tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

class MyIterableDataset(IterableDataset):

    def __init__(self, file_dir, tokenizer, block_size=128):
        self.file_dir = file_dir
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.filenames = [os.path.join(file_dir, filename) for filename in os.listdir(file_dir)]

    def parse_file(self, file_path):
        # filenames = [os.path.join(file_path, filename) for filename in os.listdir(file_path)]
        # for filename in filenames:

        skip_nexline = True
        with open(file_path, 'r') as file_obj:
            for line in file_obj:
                if not skip_nexline:
                    if not "<doc id=" in line and not "</doc>" in line:
                        if not line.isspace() and len(line) > 0:
                            batch_encoding = self.tokenizer([line],
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=self.block_size)
                            yield {"input_ids": torch.tensor(batch_encoding["input_ids"][0], dtype=torch.long)}
                if "<doc id=" in line:
                    # tile after  doc is usually title
                    skip_nexline = True
                else:
                    skip_nexline = False
    def get_steam(self,file_list):
        return chain.from_iterable(map(self.parse_file, file_list))

    def __iter__(self):
        return self.get_steam(self.filenames)

iterable_dataset = MyIterableDataset('/fsx/data/wiki_test_dir',
                                     tokenizer)

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/fsx/data/wikidemo/wiki_test",
    block_size=128,
)

# for example in train_dataset:
#     print(example)
#
for i in iterable_dataset:
    print(i)
    if i == 20:
        break

