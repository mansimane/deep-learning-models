from datasets import list_datasets
from datasets import load_dataset
datasets_list = list_datasets()
print(len(datasets_list))

for dataset in datasets_list:
    print(dataset)
# wikitext
dataset = load_dataset('wikipedia')
