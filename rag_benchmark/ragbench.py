from datasets import load_dataset

# load train/validation/test splits of individual subset
ragbench_hotpotqa = load_dataset("rungalileo/ragbench", "hotpotqa")

# load a specific split of a subset dataset
ragbench_hotpotqa = load_dataset("rungalileo/ragbench", "hotpotqa", split="test")

# load the full ragbench dataset
ragbench = {}
for dataset in [
    "covidqa",
    "cuad",
    "delucionqa",
    "emanual",
    "expertqa",
    "finqa",
    "hagrid",
    "hotpotqa",
    "msmarco",
    "pubmedqa",
    "tatqa",
    "techqa",
]:

    ragbench[dataset] = load_dataset("rungalileo/ragbench", dataset)

# load ragbench as source of data
