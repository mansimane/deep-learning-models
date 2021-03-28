from transformers import (
    AlbertTokenizer,
    LineByLineTextDataset
)

from argparse import ArgumentParser
import logging
import torch
from tqdm import tqdm
import time
import os

logger = logging.getLogger(__name__)

def main(args):
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    logger.info('Parsing training dataset ...')

    for file_name in tqdm(os.listdir(args.train_data_dir)):
        file_path = os.path.join(args.train_data_dir, file_name)
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=args.max_length)
        cur_examples = batch_encoding["input_ids"]
        cur_examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in cur_examples]
        torch.save(cur_examples, f"{args.output_dir}/{file_name.split('.')[0]}.pt")

if __name__ == "__main__":
    parser = ArgumentParser()
    # dataset
    parser.add_argument("--max_length", type=int, default=512)

    # location setting up
    parser.add_argument("--train_data_dir", default='/home/ubuntu/data/segmented/train', type=str)
    parser.add_argument("--output_dir", default='/home/ubuntu/data/segmented/pt_file', type=str)
    args = parser.parse_args()
    main(args)