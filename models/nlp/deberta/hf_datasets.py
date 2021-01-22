from datasets import list_datasets
from datasets import load_dataset
datasets_list = list_datasets()
print(len(datasets_list))
print(', '.join(dataset for dataset in datasets_list))

# wikitext
dataset = load_dataset('glue')
