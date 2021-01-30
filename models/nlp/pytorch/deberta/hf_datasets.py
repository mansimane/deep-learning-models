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

tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

class MyIterableDataset(IterableDataset):

    def __init__(self, file_path, tokenizer, block_size=128):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size

    def parse_file(self, file_path):
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

    def __iter__(self):
        return self.parse_file(self.file_path)

iterable_dataset = MyIterableDataset('/fsx/data/wikidemo/wiki_00',
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

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
config = DebertaConfig()

model = DebertaForMaskedLM(config=config)

training_args = TrainingArguments(
    output_dir="./deberta",
    overwrite_output_dir=True,

    num_train_epochs=1000,
    per_gpu_train_batch_size=2,
    learning_rate=5e-10,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e06,
    max_grad_norm=1.0,
    save_steps=10_000,
    save_total_limit=2,
    logging_first_step=False,
    logging_steps=1,
    max_steps=10000,
    gradient_accumulation_steps=10,

)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=iterable_dataset,
)

print("Starting training")
trainer.train()