import numpy as np
from house_diffusion.rplanhg_datasets import RPlanhgDataset

# Instantiate the dataset
dataset = RPlanhgDataset(set_name='train', analog_bit=False, target_set=8)

# Get the dataset length
dataset_length = len(dataset)

# Specify the output file
output_file = 'dataset_output.txt'

# Open the file in write mode and capture the print output
with open(output_file, 'w') as f:
    f.write(f"Dataset length: {dataset_length}\n")
    f.write("-" * 40 + "\n")

    # Iterate over the first few samples in the dataset
    for index in range(min(5, dataset_length)):  # Limiting to the first 5 samples
        arr, cond = dataset[index]
        graph = cond['graph']
        f.write(f"Graph data at index {index}:\n")
        f.write(f"{graph}\n")
        f.write(f"Shape of graph data at index {index}: {graph.shape}\n")
        f.write("-" * 40 + "\n")
