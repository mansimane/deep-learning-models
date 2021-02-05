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

        skip_nexline = False
        print(file_path)
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
                    # line after  doc id is usually title
                    skip_nexline = True
                else:
                    skip_nexline = False

    def get_stream(self,file_list):
        return cycle(chain.from_iterable(map(self.parse_file, file_list)))

    def __iter__(self):
        return self.get_stream(self.filenames)

# iterable_dataset = MyIterableDataset('/fsx/data/wikdemo_single',
#                                      tokenizer)
# iterable_dataset = MyIterableDataset('/fsx/data/wiki_test_dir',
#                                      tokenizer)

iterable_dataset = MyIterableDataset('/data/wikidemo/',
                                     tokenizer)
# # train_dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path="/fsx/data/wikdemo_single/wiki_00",
#     block_size=128,
# )

# for example in train_dataset:
#     print(example)
# #
# i =0
# for example in iterable_dataset:
#     print(example)
#     i += 1
#     if i == 10:
#         break

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
config = DebertaConfig()

model = DebertaForMaskedLM(config=config)

training_args = TrainingArguments(
    output_dir="./deberta",
    overwrite_output_dir=True,

    num_train_epochs=3,
    per_gpu_train_batch_size=32,
    learning_rate=1e-4,

    warmup_steps=10000,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-6,
    max_grad_norm=1.0,
    save_steps=10_000,
    save_total_limit=2,
    logging_first_step=False,
    logging_steps=1,
    max_steps=10000,
    gradient_accumulation_steps=1,

)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=iterable_dataset,
)

print("Starting training")
trainer.train()