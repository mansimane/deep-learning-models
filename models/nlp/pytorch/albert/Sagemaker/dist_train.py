import argparse
import os
import json
import subprocess as sb
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get model info')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--port', type=int, help='Port to use')
    parser.add_argument('--max_steps', type=int, help='Max steps to train')
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--platform', type=str)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=32)
    args = parser.parse_args()
    num_nodes = args.num_nodes
    platform = args.platform
    port = args.port
    max_steps = args.max_steps
    model_type = args.model_type
    gradient_accumulation_steps = args.gradient_accumulation_steps
    learning_rate = args.learning_rate
    per_gpu_train_batch_size = args.per_gpu_train_batch_size
    # environment prameter parsed from sagemaker
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]
    rank = hosts.index(current_host)
    train_data_dir = os.environ['SM_CHANNEL_TRAINING']
    validation_data_dir = os.environ['SM_CHANNEL_VALIDATION']
    finetune_data_dir = os.environ['SM_CHANNEL_FINETUNE']
    output_dir = os.environ['SM_OUTPUT_DATA_DIR']
    os.environ['NCCL_DEBUG'] = 'INFO'
    print(f'current rank is {rank}')
    if num_nodes >= 2:
        cmd = f"python -m torch.distributed.launch " \
              f"--nnodes={num_nodes} " \
              f"--node_rank={rank} " \
              f"--nproc_per_node={num_gpus} " \
              f"--master_addr={hosts[0]} " \
              f"--master_port={port} " \
              f"/opt/ml/code/train.py " \
              f"--platform {platform} " \
              f"--model_type {model_type} " \
              f"--num_nodes {num_nodes} " \
              f"--max_steps {max_steps} " \
              f"--gradient_accumulation_steps {gradient_accumulation_steps} " \
              f"--learning_rate {learning_rate} " \
              f"--per_gpu_train_batch_size {per_gpu_train_batch_size} " \
              f"--train_data_dir {train_data_dir} " \
              f"--validation_data_dir {validation_data_dir} " \
              f"--finetune_data_dir {finetune_data_dir} " \
              f"--logging_dir {output_dir} " \
              f"--output_dir {output_dir} " \
              f"--fp16 True"
    else:
        cmd = f"python /opt/ml/code/train.py " \
              f"--platform {platform} " \
              f"--model_type {model_type} " \
              f"--num_nodes {num_nodes} " \
              f"--max_steps {max_steps} " \
              f"--gradient_accumulation_steps {gradient_accumulation_steps} " \
              f"--learning_rate {learning_rate} " \
              f"--per_gpu_train_batch_size {per_gpu_train_batch_size} " \
              f"--train_data_dir {train_data_dir} " \
              f"--validation_data_dir {validation_data_dir} " \
              f"--finetune_data_dir {finetune_data_dir} " \
              f"--logging_dir {output_dir} " \
              f"--output_dir {output_dir}"
    try:
        sb.run(cmd, shell=True)
    except Exception as e:
        print(e)
    
    print('distributed script ending...')
