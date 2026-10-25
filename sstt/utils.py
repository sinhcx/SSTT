import random
import os
import torch
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_overlap_genes(adata, input_genes):

    adata_genes = set(adata.var_names)
 
    found_genes = []
    not_found_genes = []
    
    for gene in input_genes:
        if gene in adata_genes:
            found_genes.append(gene)
        else:
            not_found_genes.append(gene)
            
    if found_genes:
        print(f"Overlap genes: {found_genes}")
    else:
        print("Overlap genes: No overlap genes!")

    if not_found_genes:
        print(f"Not found genes: {not_found_genes}")
    else:
        print("Not found genes: None")
