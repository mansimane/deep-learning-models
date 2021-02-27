from argparse import ArgumentParser
from transformers import BertTokenizerFast
from datasets import load_dataset, concatenate_datasets
import time
import torch


def main(args):
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # Preprocessing the datasets: English Wikipedia and BookCorpus
    start = time.time()
    wiki_dataset = load_dataset("wikipedia", "20200501.en", split="train")
    wiki_dataset.remove_columns_("title")
    book_dataset = load_dataset("bookcorpus", split="train")
    assert wiki_dataset.features.type == book_dataset.features.type, \
        "Datasets must have same columns in order to concatenate them"
    end = time.time()
    print(f"Time to load datasets: {end - start}")

    start = time.time()
    dataset = concatenate_datasets([wiki_dataset, book_dataset])
    end = time.time()
    print(f"Time to concatenate datasets: {end - start}")

    max_seq_length = min(max_seq_length, tokenizer.model_max_length)
    print(f"initial dataset: {dataset}")

    def tokenize(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    start = time.time()
    dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=args.num_processes,
        remove_columns=["text"],
        load_from_cache_file=not args.overwrite_cache,
    )
    end = time.time()
    print(f"Time to tokenize dataset: {end - start}")
    print(f"tokenized dataset: {dataset}")

    start = time.time()
    dataset = dataset.map(
        group_texts,
        batched=True,
        num_proc=args.num_processes,
        load_from_cache_file=not args.overwrite_cache,
    )
    end = time.time()
    print(f"Time to group dataset: {end - start}")
    print(f"Grouped dataset: {dataset}")

    filename = args.data_filename
    print(f"Saving to: {filename}.pt")
    torch.save(dataset, f"{filename}.pt")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--overwrite_cache", type=bool, default=False)
    parser.add_argument("--data_filename", type=str, default="phase1_data")
    args = parser.parse_args()
    main(args)
