from datasets import list_datasets
from datasets import load_dataset
from blingfire import text_to_sentences
datasets_list = list_datasets()

print(len(datasets_list))

for dataset in datasets_list:
    print(dataset)
# wikitext
dataset = load_dataset('wikipedia')

dataset = load_dataset("wikipedia", "20200501.en", cache_dir='/run/user/1000')
# len(dataset['train']['text']) # number of documents
# 6078422

# Number of files in dataset = 13108
# Number of documents per file = 6078422/13108 = 463
shard_idx = 0
n_shards = 6
doc_idx = 0
n_docs_per_shard = 400
for shard_idx in range(n_shards):
    shard_name = '/fsx/data/wikipedia_processes_hf/wikipedia_processes_hf_' + str(shard_idx) + '.txt'
    with open(shard_name, 'w', encoding='utf-8') as out_f:
        for doc_idx in range(doc_idx, doc_idx+n_docs_per_shard):
            out_f.write(dataset['train']['text'][doc_idx])
        doc_idx += n_docs_per_shard
        print("doc_idx", doc_idx)
    print("shard_idx", shard_idx)



