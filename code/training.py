import rdkit
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
import math
import statistics
from io import StringIO
import sys
import json
from tqdm.auto import tqdm
from scipy import stats
from contextlib import redirect_stderr
from io import StringIO
import py3Dmol
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Dataset, Batch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, MFConv, GATConv, CGConv, GraphConv, GINConv
from torch_geometric.nn import TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.loader import DataLoader
from torchmetrics import R2Score, MeanAbsoluteError, MeanAbsolutePercentageError
from permetrics.regression import RegressionMetric
import plotly.graph_objects as go
import yaml
import logging
from openbabel import openbabel
from openbabel import pybel

from utils import *
from architecture import *

def start_logging():
    logging.basicConfig(filename='logs.log', level=logging.INFO)

def prepare_system():
    device = check_torch_validity()
    params = set_params_for_experiment()
    torch_geometric.seed_everything(params['SEED'])
    return device, params

def prepare_pandas_datasets(params):
    if params['CLEAN'] == False:
        dataset, clean_dataset = prepare_datasets(params)
    else:
        dataset = prepare_datasets(params)
    cond_names = renaming_and_cleaning(params, dataset)
    if params['CLEANING_FOR_TEST_SET'] == True:
        renaming_and_cleaning(params, clean_dataset)
    logging.info(f'Non-duplicated ILs: {len(list(set(dataset.smiles.values)))}')
    logging.info('Cleaning dataset')
    clean_from_problematic_and_undefined_liquids(dataset, False)
    if params['CLEANING_FOR_TEST_SET'] == True:
        logging.info('Cleaning "clean" dataset')
        clean_from_problematic_and_undefined_liquids(clean_dataset, False)
    if params['FEATURE_TRANSFORM'] == True:
        dataset['y'] = np.log(dataset['y'])
        if params['CLEANING_FOR_TEST_SET'] == True:
            clean_dataset['y'] = np.log(clean_dataset['y'])
    logging.info(f"{dataset['y'].skew() = }")
    original_dataset_len = len(dataset)
    logging.info(f"{original_dataset_len = }")
    if params['VERBOSE'] == True:
        dataset['y'].plot.hist()
    logging.info('Removing outliers from dataset')
    remove_outliers(dataset)
    if params['CLEANING_FOR_TEST_SET'] == True:
        logging.info('Removing outliers from "clean" dataset')
        remove_outliers(clean_dataset)
    y_dataset_min = dataset['y'].min()
    y_dataset_max = dataset['y'].max()
    if params['VERBOSE'] == True:
        logging.info(f"{dataset['y'].skew() = }")
        logging.info(f"{len(dataset) / original_dataset_len * 100 = }")
        dataset['y'].plot.hist()
    if params['CLEANING_FOR_TEST_SET'] == True:
        clean_dataset['y'] = (clean_dataset['y'] - dataset['y'].min()) / (dataset['y'].max() - dataset['y'].min())
    dataset['y'] = (dataset['y'] - dataset['y'].min()) / (dataset['y'].max() - dataset['y'].min())
    dataset.reset_index(inplace = True, drop = True)
    if params['VERBOSE'] == True:
        logging.info(f'Non-duplicated ILs: {len(list(set(dataset.smiles.values)))}')        
        dataset['y'].plot.hist()
    if params['CLEANING_FOR_TEST_SET'] == True:
        return dataset, clean_dataset, cond_names
    else:
        return dataset, cond_names

def prepare_pyg_datasets(params, dataset, cond_names, clean_dataset=None):
    rdBase.LogToPythonStderr()
    if (params['PREPARE_DATASET_FROM_SMI'] == True) or (params['ARCHITECTURE'] == 'two-net'):
        mol_dict, potentially_problematic_smi = prepareMolDictOfSmi(params, dataset) 
    else:
        mol_dict, potentially_problematic_smi = prepareMolDictOfMols(params, dataset)
    if len(potentially_problematic_smi) == 0:
        logging.info('No problematic molecules')
    else:
        logging.info(f'Potentially problematic molecules: {potentially_problematic_smi}')
    for cond in cond_names:
        if (params['TRANSFER'] == True) and (cond == 'P_MPa'):
            continue
        if params['CLEANING_FOR_TEST_SET'] == True: clean_dataset[cond] = (clean_dataset[cond] - dataset[cond].min()) / (dataset[cond].max() - dataset[cond].min())
        dataset[cond] = (dataset[cond] - dataset[cond].min()) / (dataset[cond].max() - dataset[cond].min())
    cond_list = dataset[cond_names].values

    if params['ARCHITECTURE'] == 'one-net':
        if params['PREPARE_DATASET_FROM_SMI'] == True:
            data = prepare_datalist_from_smi(params, dataset, cond_names, dataset.smiles.values, mol_dict=mol_dict, cond_list=cond_list, verbose=0)
        else:
            mol_list = [mol_dict[smi] for smi in dataset.smiles.values]
            data = prepare_datalist_from_mol(params, dataset, cond_names, mol_list, cond_list=cond_list)
    elif params['ARCHITECTURE'] == 'two-net':
        data = prepare_datalist_for_two_net(params, dataset, cond_names, dataset.smiles.values, mol_dict=mol_dict, cond_list=cond_list, verbose=0)
    return data

def perform_training_loop(device, params, data, cond_names, dataset, clean_dataset=None):
    embedding_size = [128,256,256,128] #  - R2 of about 0.9 #[512,1024,1024,512] - original
    linear_size = [256,128] # [1024,512]

    # conv functions - GCNConv, MFConv, GATConv, CGConv, GraphConv
    if params['ARCHITECTURE'] == 'one-net':
        model = GCN(GCNConv, input_channels = data[0].x.shape[1], embedding_size = embedding_size, linear_size = linear_size, add_params_num = len(cond_names))
    elif params['ARCHITECTURE'] == 'two-net':
        model = GCN2(GCNConv, input_channels = data[0][0].x.shape[1], embedding_size = embedding_size, linear_size = linear_size, add_params_num = len(cond_names))
    if params['TRANSFER'] == True:
        if str(device) == 'cpu':
            model.load_state_dict(torch.load('../models/base-model-density.pt', map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load('../models/base-model-density.pt'))

        if params['FINE_TUNING'] == False:
        # turn off updating some layers
            for p in model.conv1.parameters():
                p.requires_grad = False
            for p in model.conv2.parameters():
                p.requires_grad = False
            for p in model.conv3.parameters():
                p.requires_grad = False
            for p in model.conv4.parameters():
                p.requires_grad = False
    logging.info(model)
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # loss function
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # tested for well performance at 1e-3 and quite well for 5e-4
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 40)
    loss_r2 = R2Score().to(device)
    loss_mae = MeanAbsoluteError().to(device)
    loss_mare = MeanAbsolutePercentageError().to(device)

    # Use GPU for training
    model = model.to(device)
    data_size = len(data)

    # prepare smiles for test set
    if params['TARGET_FEATURE_NAME'] == 'density':
        per_test_smi = 0.002
    elif params['TARGET_FEATURE_NAME'] == 'viscosity':
        per_test_smi = 0.005
    elif params['TARGET_FEATURE_NAME'] == 'surface':
        per_test_smi = 0.01
    if params['CLEANING_FOR_TEST_SET'] == True:
        # smi_for_test = clean_dataset.sample(frac = per_test_smi, random_state=42) # many points are omitted that way but it is potentially ok to check
        smi_selected_for_test = list(set(clean_dataset.smiles.sample(frac = per_test_smi, random_state=42).values))
        smi_for_test = clean_dataset[clean_dataset['smiles'].isin(smi_selected_for_test)]
    else:
        # smi_for_test = dataset.sample(frac = per_test_smi, random_state=42) # many points are omitted that way but it is potentially ok to check
        smi_selected_for_test = list(set(dataset.smiles.sample(frac = per_test_smi, random_state=42).values))
        smi_for_test = dataset[dataset['smiles'].isin(smi_selected_for_test)]
    logging.info(f'Selected {len(list(set(smi_for_test.smiles.values)))} ILs for model testing')

    if params['ARCHITECTURE'] == 'one-net':
        loader, val_loader, test_loader, train_list, valid_list, test_list = prepare_loader_one_net(params, dataset, smi_for_test, data, cond_names)
    elif params['ARCHITECTURE'] == 'two-net':
        loader, val_loader, test_loader, train_list, valid_list, test_list, train_dataset, valid_dataset, test_dataset = prepare_loader_two_net(params, dataset, smi_for_test, data, cond_names)

    len_of_data_used = len(train_list) + len(valid_list) + len(test_list)

    logging.info(f'''
    train: {len(train_list)} \t = {len(loader)} \t loaders \t = \t {len(train_list)/(len_of_data_used)*100:.2f}% of sets = {len(train_list)/(len(data))*100:.2f}% of dataset
    valid: {len(valid_list)} \t = {len(val_loader)} \t loaders \t = \t {len(valid_list)/(len_of_data_used)*100:.2f}% of sets = {len(valid_list)/(len(data))*100:.2f}% of dataset
    test:  {len(test_list)}  \t = {len(test_loader)} \t loaders \t = \t {len(test_list)/(len_of_data_used)*100:.2f}% of sets = {len(test_list)/(len(data))*100:.2f}% of dataset
    sum to {len(train_list)/(len(data))*100 + len(valid_list)/(len(data))*100 + len(test_list)/(len(data))*100:.2f}% of dataset
    and    {len(train_list)/(len_of_data_used)*100 + len(valid_list)/ (len_of_data_used)*100 + len(test_list)/(len_of_data_used)*100:.2f} % of sets
    ''')
    train, evaluate = prepare_training_one_net(device, optimizer, model, loss_fn, loss_r2, loss_mae)
    losses, val_losses, coeffs, val_coeffs, maes, val_maes = perform_training(train, evaluate, loader, val_loader, scheduler)
    perform_validation(evaluate, loader, val_loader, test_loader)
    plot_losses(losses, val_losses, coeffs, val_coeffs)
    return model

def perform_testing(device, params, model):
    return None