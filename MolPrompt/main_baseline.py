"""Graphormer-base baseline for MoleculeNet property prediction.

Replaces MolPrompt's contrastive-pretrained model with:
  - clefourrier/graphormer-base-pcqm4mv1  (public HuggingFace checkpoint)
  - Linear classification / regression head

No proprietary checkpoints required. Runs on a single GPU (cuda:0 by default).

Usage examples
--------------
# Classification (HIV, BACE, BBBP, …)
python main_baseline.py --dataset hiv --device 0

# Regression (ESOL, FreeSolv, Lipophilicity, …)
python main_baseline.py --dataset esol --device 0 --epochs 50

# Linear probing only (freeze Graphormer backbone)
python main_baseline.py --dataset hiv --training_mode linear_probing
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import torch.multiprocessing

from Molprop_dataset.MoleculeNet_Graph import CustomDataset
from Molprop_dataset.collator_prop import collator
from Molprop_dataset.splitters import scaffold_split
from Molprop_dataset.utils import get_num_task_and_type
from Molprop_dataset.molecule_graph_model_baseline import Graph_pred_baseline

torch.multiprocessing.set_sharing_strategy('file_system')


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_classification(model, device, loader, optimizer, criterion):
    if args.training_mode == 'fine_tuning':
        model.train()
    else:
        model.eval()
    linear_model.train()
    total_loss = 0.0

    it = tqdm(loader) if args.verbose else loader
    for batch in it:
        molecule_repr, _ = model(batch)
        pred = linear_model(molecule_repr).float()

        y = torch.stack(batch.y).view(pred.shape).to(device).float()
        is_valid = y ** 2 > 0
        loss_mat = criterion(pred, (y + 1) / 2)
        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros_like(loss_mat))

        optimizer.zero_grad()
        loss = loss_mat.sum() / is_valid.sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_classification(model, device, loader):
    model.eval()
    linear_model.eval()
    y_true, y_scores = [], []

    it = tqdm(loader) if args.verbose else loader
    for batch in it:
        molecule_repr, _ = model(batch)
        pred = linear_model(molecule_repr).float()
        y = torch.stack(batch.y).view(pred.shape).to(device).float()
        y_true.append(y)
        y_scores.append(pred)

    y_true  = torch.cat(y_true,  dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            valid = y_true[:, i] ** 2 > 0
            roc_list.append(
                roc_auc_score((y_true[valid, i] + 1) / 2, y_scores[valid, i]))
        else:
            print(f'  [warn] target {i} skipped (no positive or negative samples)')

    if len(roc_list) < y_true.shape[1]:
        print(f'  Missing ratio: {1 - len(roc_list) / y_true.shape[1]:.3f}')

    return sum(roc_list) / len(roc_list), 0, y_true, y_scores


def train_regression(model, device, loader, optimizer, criterion):
    if args.training_mode == 'fine_tuning':
        model.train()
    else:
        model.eval()
    linear_model.train()
    total_loss = 0.0

    it = tqdm(loader) if args.verbose else loader
    for batch in it:
        molecule_repr, _ = model(batch)
        pred = linear_model(molecule_repr).float()
        y = torch.stack(batch.y).view(pred.shape).to(device).float()

        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_regression(model, device, loader):
    model.eval()
    linear_model.eval()
    y_true, y_pred = [], []

    it = tqdm(loader) if args.verbose else loader
    for batch in it:
        molecule_repr, _ = model(batch)
        pred = linear_model(molecule_repr).float()
        y = torch.stack(batch.y).view(pred.shape).to(device).float()
        y_true.append(y)
        y_pred.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred  = torch.cat(y_pred,  dim=0).cpu().numpy()
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}, y_true, y_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Graphormer-base baseline (no proprietary checkpoints)')
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--device',         type=int,   default=0,
                        help='CUDA device index (default: 0)')
    parser.add_argument('--training_mode',  type=str,   default='fine_tuning',
                        choices=['fine_tuning', 'linear_probing'])

    # Dataset
    parser.add_argument('--dataspace_path', type=str,   default='data')
    parser.add_argument('--dataset',        type=str,   default='hiv')
    parser.add_argument('--split',          type=str,   default='scaffold')

    # Optimisation
    parser.add_argument('--batch_size',     type=int,   default=16)
    parser.add_argument('--lr',             type=float, default=2e-5)
    parser.add_argument('--lr_scale',       type=float, default=1.0)
    parser.add_argument('--num_workers',    type=int,   default=0)
    parser.add_argument('--epochs',         type=int,   default=100)
    parser.add_argument('--weight_decay',   type=float, default=0.0)

    # Model
    parser.add_argument('--graph_hidden_dim', type=int, default=768)
    parser.add_argument('--drop_ratio',       type=float, default=0.1)

    # Output
    parser.add_argument('--eval_train',     type=int,   default=0)
    parser.add_argument('--verbose',        type=int,   default=0)
    parser.add_argument('--output_model_dir', type=str,
                        default='save_model/baseline')

    args = parser.parse_args()
    print('arguments\t', args)

    # ---- reproducibility ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f'cuda:{args.device}')
    print(f'Using device: {device}')

    # ---- dataset ----
    num_tasks, task_mode = get_num_task_and_type(args.dataset)
    dataset_folder = os.path.join(
        args.dataspace_path, 'MoleculeNet_data', args.dataset)
    data_processed_path = os.path.join(
        dataset_folder, 'processed', 'data_processed.pt')

    dataset = CustomDataset(data_processed_path)

    assert args.split == 'scaffold', 'Only scaffold split is supported'
    smiles_list = pd.read_csv(
        os.path.join(dataset_folder, 'processed', 'smiles.csv'),
        header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(
        dataset, smiles_list, null_value=0,
        frac_train=0.8, frac_valid=0.1, frac_test=0.1, pyg_dataset=False)

    _collate = partial(collator, max_node=512,
                       multi_hop_max_dist=20, spatial_pos_max=20)
    train_loader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size, num_workers=0, pin_memory=False,
        drop_last=True, collate_fn=_collate)
    val_loader = DataLoader(
        valid_dataset, shuffle=False,
        batch_size=args.batch_size, num_workers=0, pin_memory=False,
        drop_last=True, collate_fn=_collate)
    test_loader = DataLoader(
        test_dataset, shuffle=False,
        batch_size=args.batch_size, num_workers=0, pin_memory=False,
        drop_last=True, collate_fn=_collate)

    # ---- model (no proprietary checkpoint needed) ----
    model = Graph_pred_baseline(
        graph_hidden_dim=args.graph_hidden_dim,
        num_tasks=num_tasks,
        drop_ratio=args.drop_ratio,
        device_id=args.device,
    ).to(device)

    linear_model = nn.Linear(args.graph_hidden_dim, num_tasks).to(device)

    # ---- optimizer ----
    if args.training_mode == 'fine_tuning':
        param_groups = [
            {'params': model.parameters()},
            {'params': linear_model.parameters(), 'lr': args.lr * args.lr_scale},
        ]
    else:
        param_groups = [
            {'params': linear_model.parameters(), 'lr': args.lr * args.lr_scale},
        ]
    optimizer = optim.Adam(param_groups, lr=args.lr,
                           weight_decay=args.weight_decay)

    os.makedirs(args.output_model_dir, exist_ok=True)

    # ---- training loop ----
    if task_mode == 'classification':
        criterion = nn.BCEWithLogitsLoss(reduction='none')

        train_roc_list, val_roc_list, test_roc_list = [], [], []
        best_val_roc, best_val_idx = -1, 0

        for epoch in range(1, args.epochs + 1):
            loss = train_classification(model, device, train_loader, optimizer, criterion)
            print(f'Epoch: {epoch}\nLoss: {loss:.6f}')

            train_roc = 0
            if args.eval_train:
                train_roc, _, _, _ = eval_classification(model, device, train_loader)
            val_roc,  _, val_target,  val_pred  = eval_classification(model, device, val_loader)
            test_roc, _, test_target, test_pred = eval_classification(model, device, test_loader)

            train_roc_list.append(train_roc)
            val_roc_list.append(val_roc)
            test_roc_list.append(test_roc)
            print(f'train: {train_roc:.6f}\tval: {val_roc:.6f}\ttest: {test_roc:.6f}\n')

            if val_roc > best_val_roc:
                best_val_roc = val_roc
                best_val_idx = epoch - 1
                ckpt_path = os.path.join(
                    args.output_model_dir, f'{args.dataset}_model.pth')
                torch.save({'model': model.state_dict(),
                            'linear': linear_model.state_dict()}, ckpt_path)
                np.savez(
                    os.path.join(args.output_model_dir,
                                 f'{args.dataset}_evaluation.npz'),
                    val_target=val_target, val_pred=val_pred,
                    test_target=test_target, test_pred=test_pred)

        print('Best (val ROC-AUC): '
              f'train {train_roc_list[best_val_idx]:.6f}\t'
              f'val {val_roc_list[best_val_idx]:.6f}\t'
              f'test {test_roc_list[best_val_idx]:.6f}')

    else:  # regression
        criterion = nn.MSELoss()
        metric_list = ['RMSE', 'MAE']

        train_result_list, val_result_list, test_result_list = [], [], []
        best_val_rmse, best_val_idx = 1e10, 0

        for epoch in range(1, args.epochs + 1):
            loss = train_regression(model, device, train_loader, optimizer, criterion)
            print(f'Epoch: {epoch}\nLoss: {loss:.6f}')

            train_result = {'RMSE': 0, 'MAE': 0}
            if args.eval_train:
                train_result, _, _ = eval_regression(model, device, train_loader)
            val_result,  val_target,  val_pred  = eval_regression(model, device, val_loader)
            test_result, test_target, test_pred = eval_regression(model, device, test_loader)

            train_result_list.append(train_result)
            val_result_list.append(val_result)
            test_result_list.append(test_result)

            for m in metric_list:
                print(f'{m} train: {train_result[m]:.6f}\t'
                      f'val: {val_result[m]:.6f}\t'
                      f'test: {test_result[m]:.6f}')
            print()

            if val_result['RMSE'] < best_val_rmse:
                best_val_rmse = val_result['RMSE']
                best_val_idx = epoch - 1
                ckpt_path = os.path.join(
                    args.output_model_dir, f'{args.dataset}_model_best.pth')
                torch.save({'model': model.state_dict(),
                            'linear': linear_model.state_dict()}, ckpt_path)
                np.savez(
                    os.path.join(args.output_model_dir,
                                 f'{args.dataset}_evaluation_best.npz'),
                    val_target=val_target, val_pred=val_pred,
                    test_target=test_target, test_pred=test_pred)

        for m in metric_list:
            print(f'Best (RMSE), {m} '
                  f'train: {train_result_list[best_val_idx][m]:.6f}\t'
                  f'val: {val_result_list[best_val_idx][m]:.6f}\t'
                  f'test: {test_result_list[best_val_idx][m]:.6f}')
