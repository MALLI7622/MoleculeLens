"""Download and pre-process MoleculeNet datasets for main_baseline.py.

Downloads the raw CSV from OGB (for classification datasets) or from
public MoleculeNet sources (regression), then runs MolPrompt's graph
featurisation pipeline to produce the data_processed.pt files.

Usage
-----
# Prepare a single dataset (downloads raw CSV automatically via OGB)
python prepare_data.py --dataset hiv --dataspace_path data

# Multiple datasets
python prepare_data.py --dataset bace bbbp esol --dataspace_path data

Supported datasets
------------------
Classification : hiv, bace, bbbp, clintox, muv, sider, tox21, toxcast
Regression     : esol, freesolv, lipophilicity
"""

import os
import sys
import argparse
import shutil
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------
# Raw-data download helpers (use ogb where possible)
# -----------------------------------------------------------------------

OGB_DATASET_MAP = {
    'hiv':       'ogbg-molhiv',
    'bace':      None,           # not on OGB, use deepchem
    'bbbp':      None,
    'clintox':   None,
    'muv':       None,
    'sider':     None,
    'tox21':     None,
    'toxcast':   None,
    'esol':      None,
    'freesolv':  None,
    'lipophilicity': None,
}

DEEPCHEM_URLS = {
    'bbbp':      'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv',
    'bace':      'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv',
    'clintox':   'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz',
    'muv':       'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz',
    'sider':     'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz',
    'tox21':     'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
    'toxcast':   'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz',
    'esol':      'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
    'freesolv':  'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv',
    'lipophilicity': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv',
}

# Mapping from downloaded filename → smiles / label columns expected by
# MolPrompt's _load_*_dataset functions.
COLUMN_RENAMES = {
    'hiv':       {},     # already has 'smiles', 'HIV_active'
    'bbbp':      {'smiles': 'smiles', 'p_np': 'p_np'},  # BBBP.csv: smiles, name, p_np
    'bace':      {},     # bace.csv: mol, Class, …
    'esol':      {},     # delaney-processed.csv: smiles, measured log solubility in mols per litre
    'freesolv':  {},     # SAMPL.csv: iupac, smiles, expt, calc
    'lipophilicity': {},
    'tox21':     {},
    'clintox':   {},
    'muv':       {},
    'sider':     {},
    'toxcast':   {},
}


def download_ogb(dataset_name: str, raw_dir: str) -> str:
    """Download via OGB and return path to smiles+label CSV."""
    from ogb.graphproppred import GraphPropPredDataset
    ogb_name = OGB_DATASET_MAP[dataset_name]
    print(f'Downloading {dataset_name} via OGB ({ogb_name}) …')
    # OGB downloads to ~/.ogb by default; we redirect via root
    ogb_root = os.path.join(os.path.expanduser('~'), '.ogb')
    ds = GraphPropPredDataset(name=ogb_name, root=ogb_root)

    # Locate the mapping CSV (smiles + labels)
    mapping_dir = os.path.join(ogb_root, 'datasets',
                               ogb_name.replace('-', '_'), 'mapping')
    csv_candidates = [f for f in os.listdir(mapping_dir) if f.endswith('.csv.gz') or f.endswith('.csv')]
    if not csv_candidates:
        raise FileNotFoundError(f'No CSV found in {mapping_dir}')

    src = os.path.join(mapping_dir, csv_candidates[0])
    dst = os.path.join(raw_dir, csv_candidates[0])
    shutil.copy(src, dst)

    # Read, rename columns to match MolPrompt expectations, re-save
    df = pd.read_csv(dst)
    # OGB HIV: columns are 'smiles', 'HIV_active', 'mol_id'
    # MolPrompt expects exactly 'smiles' and 'HIV_active' → already correct
    clean_path = os.path.join(raw_dir, f'{dataset_name}.csv')
    df.to_csv(clean_path, index=False)
    os.remove(dst)
    print(f'  Saved raw CSV to {clean_path}')
    return clean_path


def download_deepchem(dataset_name: str, raw_dir: str) -> str:
    """Download from DeepChem S3 and return the local path."""
    import urllib.request
    url = DEEPCHEM_URLS[dataset_name]
    filename = url.split('/')[-1]
    dst = os.path.join(raw_dir, filename)
    print(f'Downloading {dataset_name} from {url} …')
    urllib.request.urlretrieve(url, dst)
    # Decompress if needed
    if dst.endswith('.gz'):
        import gzip
        out = dst[:-3]
        with gzip.open(dst, 'rb') as f_in, open(out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(dst)
        dst = out
    print(f'  Saved to {dst}')
    return dst


def get_raw_data(dataset_name: str, raw_dir: str) -> None:
    """Ensure raw CSV exists in raw_dir, downloading if necessary."""
    os.makedirs(raw_dir, exist_ok=True)
    existing = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    if existing:
        print(f'  Raw data already present: {existing}')
        return

    if OGB_DATASET_MAP.get(dataset_name):
        download_ogb(dataset_name, raw_dir)
    elif dataset_name in DEEPCHEM_URLS:
        download_deepchem(dataset_name, raw_dir)
    else:
        raise ValueError(
            f'No automatic download for dataset "{dataset_name}". '
            f'Place the raw CSV manually in {raw_dir}/')


# -----------------------------------------------------------------------
# Processing trigger (uses MolPrompt's MoleculeNetGraphDataset)
# -----------------------------------------------------------------------

def process_dataset(dataset_name: str, dataset_folder: str) -> None:
    """Run MolPrompt's graph featurisation, saving data_processed.pt."""
    processed_dir = os.path.join(dataset_folder, 'processed')
    processed_pt = os.path.join(processed_dir, 'data_processed.pt')
    if os.path.exists(processed_pt):
        print(f'  {processed_pt} already exists — skipping processing.')
        return

    os.makedirs(processed_dir, exist_ok=True)
    print(f'  Processing {dataset_name} (this may take several minutes) …')

    # MoleculeNetGraphDataset's processed_file_names property does the
    # actual graph featurisation and saves data_processed.pt as a side-
    # effect when called during __init__ → super().__init__().
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from Molprop_dataset.MoleculeNet_Graph import MoleculeNetGraphDataset
    MoleculeNetGraphDataset(dataset_folder, dataset_name)
    print(f'  Done → {processed_pt}')


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download and pre-process MoleculeNet datasets')
    parser.add_argument('--dataset', nargs='+', default=['hiv'],
                        help='One or more dataset names to prepare')
    parser.add_argument('--dataspace_path', default='data',
                        help='Root data directory (same as in main_baseline.py)')
    args = parser.parse_args()

    for ds in args.dataset:
        print(f'\n{"="*60}\nPreparing dataset: {ds}\n{"="*60}')
        dataset_folder = os.path.join(
            args.dataspace_path, 'MoleculeNet_data', ds)
        raw_dir = os.path.join(dataset_folder, 'raw')

        get_raw_data(ds, raw_dir)
        process_dataset(ds, dataset_folder)

    print('\nAll datasets ready. You can now run main_baseline.py.')
