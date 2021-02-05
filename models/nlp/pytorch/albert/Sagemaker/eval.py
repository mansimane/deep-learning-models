from transformers import (
    AlbertTokenizer,
    AlbertForPreTraining,
    LineByLineWithSOPTextDataset,
    DataCollatorForSOP,
    Trainer,
    TrainingArguments
)
from argparse import ArgumentParser
import logging
import subprocess as sb
import os
import json

logger = logging.getLogger(__name__)

def main(args):
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    logger.info('Parsing training dataset ...')
    validation_dataset = LineByLineWithSOPTextDataset(
        tokenizer=tokenizer,
        file_dir=args.validation_data_dir,
        block_size=args.max_length
    )
    logger.info('Dataset processed.')

    data_collator = DataCollatorForSOP(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )

    data_collator.tokenizer = tokenizer

    model = AlbertForPreTraining.from_pretrained(args.model_dir)

    training_args = TrainingArguments(
        per_gpu_eval_batch_size=args.per_gpu_eval_batch_size,
        logging_steps=args.logging_steps,
        local_rank=args.local_rank,
        output_dir=args.output_dir
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=validation_dataset,
        # prediction_loss_only=True,
    )

    eval_output = trainer.evaluate()
    results = {"eval_loss": eval_output["eval_loss"]}
    eval_file = os.path.join(args.output_dir, "albert_eval")
    with open(eval_file, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    # distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    # dataset
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    # model
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=20)
    # utils
    parser.add_argument("--logging_steps", type=int, default=10)
    # location setting up
    parser.add_argument("--validation_data_dir", default='/home/ubuntu/data/wiki_demo', type=str)
    parser.add_argument("--model_dir", default='./output', type=str)
    parser.add_argument("--output_dir", default='./output', type=str)

    args = parser.parse_args()
    main(args)
