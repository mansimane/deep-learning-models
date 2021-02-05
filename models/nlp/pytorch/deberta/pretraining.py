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

config = DebertaConfig()
# !wget -c https://cdn-datasets.huggingface.co/EsperBERTo/data/oscar.eo.txt

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/fsx/data/wikidemo/wiki_test",
    block_size=128,
)

# for example in train_dataset:
# ...     print(example)
# ...
# {'input_ids': tensor([    1,  4688, 13161,  1809,    16,    10,   559, 10561,     8,  2079,
#            14, 24020,    70, 28906,     6, 41654,  4620,     9, 24393,     4,
#            85, 26396,  1519,    13,     5, 34117,     9,     5,   194,    61,
#            24,  3106,     7,    28, 39028,     6, 10495,     6,     8, 11190,
#             4,     2])}

# print(example['input_ids'].shape)
# torch.Size([42])
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

model = DebertaForMaskedLM(config=config)

training_args = TrainingArguments(
    output_dir="./deberta",
    overwrite_output_dir=True,

    num_train_epochs=1000,
    per_gpu_train_batch_size=32,
    learning_rate=5e-10,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e06,
    max_grad_norm=1.0,
    save_steps=10_000,
    save_total_limit=2,
    logging_first_step=True,
    logging_steps=1,

)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)
trainer.train()