from transformers import (
    AlbertConfig,
    AlbertTokenizer,
    AlbertForPreTraining,
    LineByLineWithSOPTextDataset,
    DataCollatorForSOP,
    Trainer,
    TrainingArguments
)
from argparse import ArgumentParser
import logging
import os
import json
import torch

import smdistributed.dataparallel.torch.distributed as dist

logger = logging.getLogger(__name__)

def main(args):
    albert_base_configuration = AlbertConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
    )

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    logger.info('Parsing training dataset ...')
    train_dataset = LineByLineWithSOPTextDataset(
        tokenizer=tokenizer,
        file_dir=args.train_data_dir,
        block_size=args.max_length
    )

    logger.info('Dataset processed.')

    data_collator = DataCollatorForSOP(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )
    data_collator.tokenizer = tokenizer

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
        logging_dir=f"{args.output_dir}/logs",
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        local_rank=args.local_rank,
        dataloader_num_workers=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )
    logger.info('Trainer initialing, start training...')
    train_output = trainer.train()
    logger.info(f'Saving model to {args.output_dir}...')
    if trainer.is_world_process_zero():
        # save model and vocab files
        trainer.save_model(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        train_res = os.path.join(args.output_dir, "albert_train_res")
        with open(train_res, 'w') as f:
            json.dump(train_output, f)
    print('end of a single process')

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
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--fp16", type=bool, default=True)
    # utils
    parser.add_argument("--save_steps", type=int, default=130000)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--overwrite_output_dir", type=bool, default=True)
    # location setting up
    parser.add_argument("--train_data_dir", default=os.environ['SM_CHANNEL_TRAINING'], type=str)
    parser.add_argument("--output_dir", default=os.environ['SM_OUTPUT_DATA_DIR'], type=str)

    args = parser.parse_args()
    args.local_rank = dist.get_local_rank()
    main(args)
