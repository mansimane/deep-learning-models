from transformers import (
    AlbertConfig,
    AlbertTokenizer,
    AlbertForPreTraining,
    LineByLineTextDataset,
    LineByLineWithSOPTextDataset,
    DataCollatorForSOP,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from argparse import ArgumentParser
import logging
import time
import subprocess as sb
import psutil
import os
import io
import json

import torch
from torch.utils.data import IterableDataset
from awsio.python.lib.io.s3.s3dataset import S3IterableDataset

logger = logging.getLogger(__name__)

class s3_dataset(IterableDataset):

    def __init__(self, urls):
        self.urls = urls
        self.dataset = S3IterableDataset(urls, shuffle_urls=True)

    def data_generator(self):
        try:
            while True:
                filename, fileobj = next(self.dataset_iter)
                examples = torch.load(io.BytesIO(fileobj))
                for example in examples:
                    yield example

        except StopIteration as e:
            print(e)
            self.dataset = S3IterableDataset(self.urls, shuffle_urls=True)
            return self.__iter__()

    def __iter__(self):
        self.dataset_iter = iter(self.dataset)
        return self.data_generator()

def main(args):
    albert_base_configuration = AlbertConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
    )

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    logger.info('Parsing training dataset ...')

    bucket = "s3://yuliu-dev-east/wiki_bookcorpus_demo"
    # load pre-processed pt files, max_len=512
    train_dataset = s3_dataset(bucket)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )

    model = AlbertForPreTraining(config=albert_base_configuration)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        max_steps=args.max_steps,
        per_gpu_train_batch_size=args.per_gpu_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        local_rank=args.local_rank
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )
    logger.info('Trainer initialed, start training...')
    start = time.time()
    trainer.train()
    duration = time.time() - start
    print(duration)


if __name__ == "__main__":
    parser = ArgumentParser()
    # distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--platform", type=str)
    # dataset
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    # model
    parser.add_argument('--model_type', type=str, default='albert_base')
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--fp16", type=bool, default=False)
    # utils
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--overwrite_output_dir", type=bool, default=True)
    # location setting up
    parser.add_argument("--train_data_dir", default='/home/ubuntu/data/wiki_demo', type=str)
    parser.add_argument("--validation_data_dir", default='/root/data/wiki_demo/wiki_00', type=str)
    parser.add_argument("--finetune_data_dir", default='/home/ubuntu/data/squad', type=str)
    parser.add_argument("--logging_dir", default='./log', type=str)
    parser.add_argument("--output_dir", default='./output', type=str)

    args = parser.parse_args()
    main(args)
