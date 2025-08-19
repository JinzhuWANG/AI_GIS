import pandas as pd
from datasets import load_dataset


# Specify where to save the data https://huggingface.co/datasets/blanchon/UC_Merced
UC_Merced = load_dataset(
    "blanchon/UC_Merced", 
    cache_dir="data/UC_Merced"
)


