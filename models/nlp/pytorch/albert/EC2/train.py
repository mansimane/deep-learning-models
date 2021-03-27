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
import time
import subprocess as sb
import psutil
import os
import json
import torch

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

    # logger.info('Dataset processed.')
    #
    # data_collator = DataCollatorForSOP(
    #     tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    # )
    # data_collator.tokenizer = tokenizer
    #
    # model = AlbertForPreTraining(config=albert_base_configuration)
    #
    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     overwrite_output_dir=args.overwrite_output_dir,
    #     max_steps=args.max_steps,
    #     per_gpu_train_batch_size=args.per_gpu_train_batch_size,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     save_steps=args.save_steps,
    #     save_total_limit=args.save_total_limit,
    #     logging_steps=args.logging_steps,
    #     logging_dir=args.logging_dir,
    #     learning_rate=args.learning_rate,
    #     fp16=args.fp16,
    #     local_rank=args.local_rank
    # )
    #
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=train_dataset
    # )
    # logger.info('Trainer initialed, start training...')
    # t_start = time.time()
    # output = trainer.train()
    # print(output)
    # t_end = time.time()
    # train_time = t_end - t_start
    # logger.info(f'Training time {round(train_time / 60)} mins')
    # logger.info(f'Saving model to {args.output_dir}...')
    # # save model and vocab files
    # trainer.save_model(args.output_dir)
    # tokenizer.save_vocabulary(args.output_dir)
    # # start validation
    # # clear cache for squad fine-tune
    # torch.cuda.empty_cache()
    # if trainer.is_world_process_zero():
    #     print('master process located.')
    #     # print('Evaluation starting...')
    #     cmd = f"python eval.py " \
    #           f"--per_gpu_eval_batch_size {args.per_gpu_train_batch_size} " \
    #           f"--validation_data_dir {args.validation_data_dir} " \
    #           f"--output_dir {args.output_dir} " \
    #           f"--model_dir {args.output_dir}"
    #     try:
    #         sb.run(cmd, shell=True)
    #     except Exception as e:
    #         print(e)
    #     print('Evaluation Ended...')
    #     # Run squad fine-tuning locally
    #     print('Fine tuning start...')
    #     cmd = f"python run_qa.py \
    #                       --model_name_or_path {args.output_dir} \
    #                       --dataset_name squad \
    #                       --do_train \
    #                       --do_eval \
    #                       --per_device_train_batch_size 12 \
    #                       --learning_rate 3e-5 \
    #                       --num_train_epochs 2 \
    #                       --max_seq_length 384 \
    #                       --doc_stride 128 \
    #                       --output_dir ./output/squad_res/"
    #     try:
    #         sb.run(cmd, shell=True)
    #     except Exception as e:
    #         print(e)
    #     torch.cuda.empty_cache()
    #     print('Fine tuning Ended...')
    # print('end of a single process')

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
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=20)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--fp16", type=bool, default=False)
    # utils
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--overwrite_output_dir", type=bool, default=True)
    # location setting up
    parser.add_argument("--train_data_dir", default='/home/ubuntu/data/wiki_demo', type=str)
    parser.add_argument("--validation_data_dir", default='/home/ubuntu/data/wiki_demo', type=str)
    parser.add_argument("--finetune_data_dir", default='/home/ubuntu/data/squad', type=str)
    parser.add_argument("--logging_dir", default='./log', type=str)
    parser.add_argument("--output_dir", default='./output', type=str)

    args = parser.parse_args()
    main(args)
