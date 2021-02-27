import logging
import os
import sys
from dataclasses import dataclass, field
import math

from datasets import load_dataset
import torch # to load dataset

import transformers
from transformers import (
    BertConfig,
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    get_polynomial_decay_schedule_with_warmup,
)
from apex.optimizers import FusedLAMB
from transformers.trainer_utils import get_last_checkpoint, is_main_process


logger = logging.getLogger(__name__)


def create_optimizer_and_scheduler(model, args):
    m_params = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in m_params if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in m_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = FusedLAMB(optimizer_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer,
                                                             num_warmup_steps=math.ceil(args.max_steps * args.warmup_ratio),
                                                             num_training_steps=args.max_steps,
                                                             power=0.5)
    return optimizer, lr_scheduler


@dataclass
class DataTrainingArguments:

    train_dataset: str = field(
        default="phase1_data.pt",
        metadata={
            "help": "Where preprocess_data.py saved the dataset"
        }
    )

    validation_split_percentage: float = field(
        default=0.15,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )

    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    phase2: bool =  field(
        default=False,
        metadata={
            "help": "False if running phase 1, True if running phase 2"
        }
    )

    phase1_output_dir: str = field (
        default="./bert_phase1/"
    )


def main():
    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # set config, tokenizer, and model
    config = BertConfig()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    model = BertForMaskedLM(config=config)
    if data_args.phase2:
        model_state = torch.load(os.path.join(data_args.phase1_output_dir, "pytorch_model.bin"))
        model.load_state_dict(model_state)
        logger.info(f"Loaded model from phase 1, continue pre-training")

    # Load preprocessed datasets
    datasets = torch.load(data_args.train_dataset)
    datasets = datasets.train_test_split(test_size=data_args.validation_split_percentage)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=create_optimizer_and_scheduler(model, training_args),
    )

    # Training
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
    if trainer.is_world_process_zero():
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))


if __name__ == "__main__":
    main()
