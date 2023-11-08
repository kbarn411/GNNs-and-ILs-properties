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

def check_torch_validity():
    """
    Checks the validity of the Torch library by performing several operations.

    This function checks the availability of a CUDA device and sets the device to be
    used by Torch accordingly. It also prints the device being used by Torch, the
    version of Torch, and the version of CUDA used for Torch compilation. Additionally,
    it attempts to perform a CUDA operation using `torch.zeros(1).cuda()` and catches
    any potential runtime errors.

    Returns:
        device (torch.device): The device used by Torch, either "cuda" or "cpu".

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device used by torch: {device}')
    logging.info(f'Version of torch: {torch.__version__}')
    logging.info(f'CUDA used for torch compilation: {torch.version.cuda}')
    try:
        logging.info(f'{torch.zeros(1).cuda()}')
    except RuntimeError as inst:
        logging.info(f'Runtime Error {inst}')
    return device

def set_params_for_experiment(yaml_params_path='./params.yaml'):
    """
    Set the parameters for the experiment.

    Args:
        yaml_params_path (str, optional): The path to the YAML file containing the parameters. Defaults to './params.yaml'.

    Returns:
        dict: A dictionary containing the parameters.
    """
    params = yaml.load(open(yaml_params_path), Loader=yaml.loader.SafeLoader)
    return params

def prepare_datasets(params):
    """
    Prepare the datasets for training and testing.

    Parameters:
        params (dict): A dictionary containing the parameters for dataset preparation.

    Returns:
        dataset (pd.DataFrame): The dataset prepared for training or testing.
        clean_dataset (pd.DataFrame, optional): The cleaned dataset for testing, if applicable.
    """
    if params['CLEAN'] == True:
        dataset = pd.read_csv(f'../data/data-{params["TARGET_FEATURE_NAME"]}-clean.csv', sep=',')
        params['CLEANING_FOR_TEST_SET'] = False
    else:
        dataset = pd.read_csv(f'../data/data-{params["TARGET_FEATURE_NAME"]}.csv', sep=',')
        params['CLEANING_FOR_TEST_SET'] = True
    if params['CLEANING_FOR_TEST_SET'] == True:
        clean_dataset = pd.read_csv(f'../data/data-{params["TARGET_FEATURE_NAME"]}-clean.csv', sep=',')
    if params['CLEANING_FOR_TEST_SET'] == True:
        return dataset, clean_dataset
    else:
        return dataset
    
def renaming_and_cleaning(params, dataset):
    """
    Renames target features and cleans the dataset based on the given parameters.

    Args:
        params (dict): A dictionary containing the parameters for renaming and cleaning.
        dataset (pandas.DataFrame): The dataset to be renamed and cleaned.

    Returns:
        list: A list of condition names based on the target feature name.

    Raises:
        None
    """
    if params["TARGET_FEATURE_NAME"] == 'density':
        dataset.rename(columns={"d_kg*m-3": "y"}, inplace=True)
        cond_names = ['T_K', 'P_MPa']
    elif params["TARGET_FEATURE_NAME"] == 'viscosity':
        dataset.rename(columns={"n_mPas": "y"}, inplace=True)
        if params["TRANSFER"] == False:
            cond_names = ['T_K']
        else:
            dataset['P_MPa'] = 0.1
            cond_names = ['T_K', 'P_MPa']
    elif params["TARGET_FEATURE_NAME"] == 'surface':
        dataset.rename(columns={"s_mNm": "y"}, inplace=True)
        if params["TRANSFER"] == False:
            cond_names = ['T_K']
        else:
            dataset['P_MPa'] = 0.1
            cond_names = ['T_K', 'P_MPa']

    # droping columns with no value other than accountancy
    dataset.drop(['IL', 'cation', 'anion'], axis = 1, inplace=True)
    dataset.dropna(inplace=True)
    # drop duplicates
    dataset.drop_duplicates(inplace=True)
    return cond_names

def remove_outliers(dataset, outlier_method='MAD'):
    """
    Remove outliers from the dataset using the specified outlier detection method.

    Parameters:
        dataset (pandas.DataFrame): The dataset containing the outliers.
        outlier_method (str): The method to be used for outlier detection. 
            Available options: 'Z_score', 'IQR', 'log_IQR', 'MAD', '' (default).

    Returns:
        None

    Raises:
        Exception: If an unavailable option is specified for the outlier_method.

    Notes:
        - This function modifies the dataset in-place by dropping the outlier rows.
        - The outlier detection methods are as follows:
            - Z_score: Uses the Z-score method to detect outliers.
            - IQR: Uses the Interquartile Range (IQR) method to detect outliers.
            - log_IQR: Uses the log-transformed version of the IQR method to detect outliers.
            - MAD: Uses the Median Absolute Deviation (MAD) method to detect outliers.
            - '': Does not perform any outlier detection.

    """
    if outlier_method == 'Z_score':
        # outlier detection using Z-score
        threshold = 3
        for _ in range(1):
            to_drop = dataset[(np.abs(stats.zscore(dataset['y'])) > threshold)]
            dataset.drop(to_drop.index, axis = 0, inplace = True)
        logging.info(f'Dropped {to_drop.shape[0]} outliers')

    elif outlier_method == 'IQR':
        # outlier detection using IQR
        Q1, Q3 = np.percentile(dataset['y'], [25,75])
        ul = Q3 + 1.5 * (Q3 - Q1)
        ll = Q1 - 1.5 * (Q3 - Q1)
        to_drop = dataset[(dataset['y'] < ll) | (dataset['y'] > ul)]
        dataset.drop(to_drop.index, axis = 0, inplace = True)
        logging.info(f'Dropped {to_drop.shape[0]} outliers')

    elif outlier_method == 'log_IQR':
        # outlier detection using log version of IQR
        Q1,Q3 = np.percentile(np.log(dataset['y']), [25,75])
        ul = Q3 + 1.5 * (Q3 - Q1)
        ll = Q1 - 1.5 * (Q3 - Q1)
        to_drop = dataset[(np.log(dataset['y']) < ll) | (np.log(dataset['y']) > ul)]
        dataset.drop(to_drop.index, axis = 0, inplace = True)
        logging.info(f'Dropped {to_drop.shape[0]} outliers')

    elif outlier_method == 'MAD':
        # outlier detection using median absolute deviation method (MAD)
        threshold = 3
        med = np.median(dataset['y'])
        mad = np.abs(stats.median_abs_deviation(dataset['y']))
        to_drop = dataset[((dataset['y'] - med) / mad) > threshold]
        dataset.drop(to_drop.index, axis = 0, inplace = True)
        logging.info(f'Dropped {to_drop.shape[0]} outliers')
        
    elif outlier_method == '':
        logging.info('No outlier detection method.')
    else:
        raise Exception('Sorry! Unavailable option')

def clean_from_problematic_and_undefined_liquids(dataset, verbose):
    """
    Removes problematic and undefined liquids from the dataset.
    
    Parameters:
        dataset (DataFrame): The dataset containing the liquids.
        verbose (bool): If True, print the deleted smiles.
    
    Returns:
        None
    """
    for smile in dataset.smiles.values:
        temp = smile.split('.')
        # check for multicationic liquids and remove them
        if len(temp) > 2:
            dataset.drop(dataset[dataset['smiles'] == smile].index, inplace = True)
            if verbose: 
                logging.info(f'deleting {smile}')
        # check for possible nan in smiles
        if 'nan' in temp:
            dataset.drop(dataset[dataset['smiles'] == smile].index, inplace = True)
            if verbose: 
                logging.info(f'deleting {smile}')

def get_atom_features(params, mol, return_type="numpy"):
    """
    Calculates the atom features of a molecule.

    Parameters:
        params (dict): A dictionary containing the parameters.
        mol (Chem.Mol): The molecule to be processed.
        return_type (str): The type of return. Available options: 'numpy', 'torch'.

    Returns:
        atom_features (np.ndarray or torch.Tensor): The atom features of the molecule.
    """
    if params['REMOVE_HS'] == True: 
        mol = Chem.RemoveHs(mol)
    atomic_number = []
    if not params['REMOVE_NUM_HS']: 
        num_hs = []
    hybr = []
    charges = []
    aromacity = []
    degrees = []

    if params['ENGINE'] == 'OB':
        mol_mol = Chem.MolToMolBlock(mol)
        bel_mol = pybel.readstring("mol", mol_mol)
        if params['CHARGE_MODEL'] != 'formal':
            ob_charge_model = openbabel.OBChargeModel.FindType(params['CHARGE_MODEL'])
            ob_charge_model.ComputeCharges(bel_mol.OBMol)
    elif params['ENGINE'] == 'RDKIT':
        AllChem.ComputeGasteigerCharges(mol)
    else:
        raise Exception('Sorry! Unavailable option')

    for i, atom in enumerate(mol.GetAtoms()):
        atomic_number.append(atom.GetAtomicNum())
        if not params['REMOVE_NUM_HS']: 
            num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))
        hybr.append(atom.GetHybridization())

        if params['ENGINE'] == 'OB':
            if params['CHARGE_MODEL'] != 'formal': charge = bel_mol.OBMol.GetAtom(i+1).GetPartialCharge()
            else: charge = bel_mol.OBMol.GetAtom(i+1).GetFormalCharge()
        elif params['ENGINE'] == 'RDKIT':
            charge = atom.GetDoubleProp("_GasteigerCharge")
        else:
            raise Exception('Sorry! Unavailable option')

        if math.isnan(charge):
            charge = 0
        if params['CHARGE_MODEL'] != '': charges.append(charge)

        aromacity.append(atom.GetIsAromatic())
        degrees.append(atom.GetDegree())

    le = LabelEncoder()
    hybr = le.fit_transform(hybr)

    if return_type == 'torch':
        if params['REMOVE_NUM_HS'] == False:
          return torch.tensor([atomic_number, num_hs, hybr, charges, aromacity, degrees]).t()
        else:
          return torch.tensor([atomic_number, hybr, charges, aromacity, degrees]).t()
    elif return_type == 'numpy':
        if params['REMOVE_NUM_HS'] == False:
          result = np.array([atomic_number, num_hs, hybr, charges, aromacity, degrees])
        else:
          result = np.array([atomic_number, hybr, charges, aromacity, degrees])
        return np.transpose(result)

def get_edges_info(params, mol, return_type="numpy"):
    """
    Calculates the edges features for bonds in a molecule.

    Parameters:
        params (dict): A dictionary containing the parameters.
        mol (Chem.Mol): The molecule to be processed.
        return_type (str): The type of return. Available options: 'numpy', 'torch'.

    Returns:
        edge_features (np.ndarray or torch.Tensor): The edge features of the molecule.
    """
    if params['REMOVE_HS'] == True: 
        mol = Chem.RemoveHs(mol)
    row, col, bonds_types = [], [], []

    for i, bond in enumerate(mol.GetBonds()):
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bonds_types += [bond.GetBondTypeAsDouble(), bond.GetBondTypeAsDouble()]

    if (params['PREPARE_LINKING_IONIC_BOND'] == True) and (params['PREPARE_DATASET_FROM_SMI'] == False):
        mol_mol = Chem.MolToMolBlock(mol)
        bel_mol = pybel.readstring("mol", mol_mol)

        ionic_bond_test = 0
        for i, _ in enumerate(mol.GetAtoms()):
            formal_charge = bel_mol.OBMol.GetAtom(i+1).GetFormalCharge()
            if formal_charge > 0:
                start = i
                ionic_bond_test += 1
            elif formal_charge < 0:
                end = i
                ionic_bond_test += 1
        row += [start, end]
        col += [end, start]
        bonds_types += [-1, -1]
        if ionic_bond_test != 2:
            logging.info(f'Issue {ionic_bond_test = }, mol = {Chem.MolToSmiles(mol)}')

    if return_type == 'torch':
        return torch.tensor([row, col], dtype=torch.long), torch.tensor(bonds_types, dtype=torch.float)
    elif return_type == 'numpy':
        return np.array([row, col], dtype=np.int_), np.array(bonds_types, dtype=np.float32)

def prepare_datalist_from_mol(params, dataset, cond_names, mol_list, cond_list=None, input_type='numpy'):
    """
    Prepare list of data from a list of molecules. 
    Usfeul for 1 IL - 1 molecule - 1 graph scenario.

    Parameters:
        dataset (pandas.DataFrame): The dataset to be prepared.
        cond_names (list): The list of conditions' types to be used.
        mol_list (list): The list of molecules.
        cond_list (list): The list of conditions.
        input_type (str): The type of input.

    Returns:
        data_list (list): The prepared list of data.
    """
    data_list = []

    for i, mol in enumerate(mol_list):

        x = get_atom_features(params, mol, return_type='torch')
        edge_index, edge_weights = get_edges_info(params, mol, return_type='torch')
        y = torch.tensor(dataset.y.values[i], dtype=torch.float).view([1, 1])

        data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr = edge_weights, y=y)
        if cond_list is not None:
            data.cond = torch.tensor(np.array(cond_list[i])).view(-1,len(cond_names)).to(torch.float32)
        data.smi = dataset.smiles.values[i]
        data_list.append(data)

    return data_list

def prepare_datalist_from_smi(params, dataset, cond_names, smi_list, mol_dict, cond_list=None, input_type='numpy', verbose=1):
    """
    Prepare list of data from a list of smiles for separate ions. 
    Usfeul for 1 IL - 2 molecules - 1 graph scenario.

    Parameters:
        params (dict): A dictionary containing the parameters.
        dataset (pandas.DataFrame): The dataset to be prepared.
        cond_names (list): The list of conditions' types to be used.
        smi_list (list): The list of molecules.
        cond_list (list): The list of conditions.
        input_type (str): The type of input.
        verbose (int): The verbosity.
    
    Returns:
        data_list (list): The prepared list of data.
    """
    data_list = []

    if input_type != 'numpy':
        raise Exception("currently torch option not available for smi based features")

    for i, smi in enumerate(smi_list):
        c_smi, a_smi = smi.split('.')
        if verbose: logging.info(smi.split('.'))
        c_mol = mol_dict[c_smi]
        a_mol = mol_dict[a_smi]

        #calculate x for c and a
        c_x = get_atom_features(params, c_mol)
        a_x = get_atom_features(params, a_mol)

        temp_up = np.concatenate([c_x, np.zeros([c_x.shape[0],a_x.shape[1]])], axis=1)
        temp_down = np.concatenate([np.zeros([a_x.shape[0],c_x.shape[1]]), a_x], axis=1)
        b = np.concatenate([temp_up, temp_down]) #b for both c and a
        x = torch.tensor(b)
        if verbose: logging.info(c_x.shape, a_x.shape, x.shape)

        #calculate edges for c and a
        c_edge_index, c_edge_weights = get_edges_info(params, c_mol)
        a_edge_index, a_edge_weights = get_edges_info(params, a_mol)

        b = np.concatenate([c_edge_index, a_edge_index], axis=1)
        if (params['PREPARE_LINKING_IONIC_BOND'] == True) and (params['PREPARE_DATASET_FROM_SMI'] == True):
            b_0 = np.append(b[0], [0, x.shape[0]-1]) #old: row
            b_1 = np.append(b[1], [x.shape[0]-1, 0]) #old: col
            b = np.array([b_0, b_1])
        edge_index = torch.tensor(b)
        if verbose: logging.info(c_edge_index.shape, a_edge_index.shape, edge_index.shape)

        b = np.concatenate([c_edge_weights, a_edge_weights])
        if (params['PREPARE_LINKING_IONIC_BOND'] == True) and (params['PREPARE_DATASET_FROM_SMI'] == True):
            b = np.concatenate([b, [-1, -1]])
        edge_weights = torch.tensor(b)
        if verbose: logging.info(c_edge_weights.shape, a_edge_weights.shape, edge_weights.shape)

        y = torch.tensor(dataset.y.values[i], dtype=torch.float).view([1, 1])

        data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr = edge_weights, y=y)
        if cond_list is not None:
            data.cond = torch.tensor(np.array(cond_list[i])).view(-1,len(cond_names)).to(torch.float32)
        data.smi = dataset.smiles.values[i]
        data_list.append(data)

    return data_list

def prepare_datalist_for_two_net(params, dataset, cond_names, smi_list, mol_dict, cond_list=None, input_type='numpy', verbose=1):
    """
    Prepare list of data from a list of smiles for separate ions. 
    Usfeul for 1 IL - 2 molecules - 2 graphs scenario (ie. "two-net" schema).

    Parameters:
        params (dict): A dictionary containing the parameters.
        dataset (pandas.DataFrame): The dataset to be prepared.
        cond_names (list): The list of conditions' types to be used.
        smi_list (list): The list of molecules.
        cond_list (list): The list of conditions.
        input_type (str): The type of input.
        verbose (int): The verbosity.
    
    Returns:
        data_list (list): The prepared list of data.
    """
    data_list = []

    if input_type != 'numpy':
        raise Exception("currently torch option not available for smi based features")

    for i, smi in enumerate(smi_list):
        c_smi, a_smi = smi.split('.')
        if verbose: logging.info(smi.split('.'))
        c_mol = mol_dict[c_smi]
        a_mol = mol_dict[a_smi]

        #calculate x for c and a
        c_x = get_atom_features(params, c_mol, return_type='torch')
        a_x = get_atom_features(params, a_mol, return_type='torch')

        #calculate edges for c and a
        c_edge_index, c_edge_weights = get_edges_info(params, c_mol, return_type='torch')
        a_edge_index, a_edge_weights = get_edges_info(params, a_mol, return_type='torch')

        y = torch.tensor(dataset.y.values[i], dtype=torch.float).view([1, 1])

        data_c = torch_geometric.data.Data(x=c_x, edge_index=c_edge_index, edge_attr = c_edge_weights, y=y)
        data_a = torch_geometric.data.Data(x=a_x, edge_index=a_edge_index, edge_attr = a_edge_weights, y=y)

        if cond_list is not None:
            data_c.cond = torch.tensor(np.array(cond_list[i])).view(-1,len(cond_names)).to(torch.float32)
            data_a.cond = torch.tensor(np.array(cond_list[i])).view(-1,len(cond_names)).to(torch.float32)
        data_c.smi = dataset.smiles.values[i]
        data_a.smi = dataset.smiles.values[i]
        data_list.append([data_c, data_a])

    return data_list

class PairDataset(torch.utils.data.Dataset):
    """
    A class for pair dataset (useful for 1 IL - 2 molecules - 2 graphs scenario).
    """
    def __init__(self, data_list):
        self.datasetC = []
        self.datasetA = []
        for data in data_list:
            self.datasetC.append(data[0])
            self.datasetA.append(data[1])

    def __getitem__(self, idx):
        return self.datasetC[idx], self.datasetA[idx]

def collate(self, data_list):
    """
    Supporting function for dataloader for 1 IL - 2 molecules - 2 graphs scenario.
    """
    batchC = Batch.from_data_list([data[0] for data in data_list])
    batchA = Batch.from_data_list([data[1] for data in data_list])
    return batchC, batchA

def OBoptimize(smi):
    gen3d = openbabel.OBOp.FindType("gen3D")
    bel_mol = pybel.readstring("smi", smi)
    bel_mol.addh()
    gen3d.Do(bel_mol.OBMol, "--best")

    # optimization
    bel_mol.localopt()

    # conversion to mol
    mol_mol = bel_mol.write("mol")
    return mol_mol

def RDOptimize(params, smi, track_problems=False):
    if track_problems: potentially_problematic_one_smi = False
    mol = Chem.MolFromSmiles(smi)
    with StringIO() as buf:
        with redirect_stderr(buf):
            mol_h = Chem.AddHs(mol)
            res_molh_error = buf.getvalue()
            if res_molh_error != '':
                logging.info(res_molh_error, smi, 'issue during adding Hs', sep=' ')
                if track_problems: potentially_problematic_one_smi = True
    try:
        ps = AllChem.ETKDGv3()
        ps.randomSeed = params['SEED']
        ps.useRandomCoords = True
        ps.maxAttempts = 50
        with StringIO() as buf:
            with redirect_stderr(buf):
                cids = AllChem.EmbedMultipleConfs(mol_h, numConfs = 10, params = ps)
                res_cids_error = buf.getvalue()
                if res_cids_error != '':
                    logging.info(res_cids_error, smi, 'issue during multiple embedding', sep=' ')
                    if track_problems: potentially_problematic_one_smi = True
            results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters = 500)
        if len(results) == 0:
            with StringIO() as buf:
                with redirect_stderr(buf):
                    res_embedding = AllChem.EmbedMolecule(mol_h, useRandomCoords=True, randomSeed=params['SEED'])
                    res_embedding_error = buf.getvalue()
                    logging.info(res_embedding_error, smi, 'issue after optim', sep=' ')
                    if track_problems: potentially_problematic_one_smi = True
                    if res_embedding_error != '':
                        logging.info(res_embedding_error, smi, 'issue with single embedding - try to compute 2D', sep=' ')
                        AllChem.Compute2DCoords(mol_h) # to provide at least 2D coords - OB performs better anyway
            final_molecule = mol_h
        else:
            min_energy, min_energy_index = 10000, 0
            for index, result in enumerate(results):
                if(min_energy>result[1]):
                    min_energy = result[1]
                    min_energy_index = index
            final_molecule = Chem.Mol(mol_h, False, min_energy_index)
    except ValueError as veinst:
        with StringIO() as buf:
            with redirect_stderr(buf):
                res_embedding = AllChem.EmbedMolecule(mol_h, useRandomCoords=True, randomSeed=params['SEED'])
                res_embedding_error = buf.getvalue()
                if res_embedding_error != '':
                    logging.info(veinst, res_embedding_error, smi, 'issue with ValueError - not very known reason', sep=' ')
                    if track_problems: potentially_problematic_one_smi = True
        final_molecule = mol_h
    if track_problems:
        return final_molecule, potentially_problematic_one_smi
    else:
        return final_molecule


def prepareMolDictOfSmi(params, dataset):
    smi_list = list(set(dataset.smiles.values))
    mol_dict = {}
    smi_set_ions = set()
    for i in range(len(smi_list)):
        smis = smi_list[i]
        smis = smis.split('.')
        smi_set_ions.add(smis[0])
        smi_set_ions.add(smis[1])
    smi_list_ions = list(smi_set_ions)
    potentially_problematic_smi = set()
    for i in tqdm(range(len(smi_list_ions))):
        smi = smi_list_ions[i]
        if params['ENGINE'] == 'OB':
            mol_mol = OBoptimize(smi)
            try:
                mol_dict[smi] = Chem.MolFromMolBlock(mol_mol)
            except:
                logging.info(smi, 'issue with MolFromMol2Block', sep=' ')
                potentially_problematic_smi.add(smi)
                mol = Chem.MolFromSmiles(smi)
                mol_h = Chem.AddHs(mol)
                mol_dict[smi] = mol_h
        elif params['ENGINE'] == 'RDKIT':
            mol_dict[smi], potentially_problematic_one_smi = RDOptimize(smi, track_problems = True)
            if potentially_problematic_one_smi: potentially_problematic_smi.add(smi)
    return mol_dict, potentially_problematic_smi

def prepareMolDictOfMols(params, dataset):
    smi_list = list(set(dataset.smiles.values))
    mol_dict = {}
    potentially_problematic_smi = set()
    for i in tqdm(range(len(smi_list))):
        smi = smi_list[i]
        if params['ENGINE'] == 'OB':
            mol_mol = OBoptimize(smi)
            try:
                mol_dict[smi] = Chem.MolFromMolBlock(mol_mol)
            except:
                logging.info(smi, 'issue with MolFromMolBlock', sep=' ')
                potentially_problematic_smi.add(smi)
                mol = Chem.MolFromSmiles(smi)
                mol_h = Chem.AddHs(mol)
                mol_dict[smi] = mol_h
        elif params['ENGINE'] == 'RDKIT':
            mol_dict[smi], potentially_problematic_one_smi = RDOptimize(smi, track_problems = True)
            if potentially_problematic_one_smi: potentially_problematic_smi.add(smi)
    return mol_dict, potentially_problematic_smi

def prepare_loader_one_net(params, dataset, smi_for_test, data, cond_names):
    if params['SPLITTER'] == 'scaffold':
        per_train = 0.8
        smi_list = []
        for smile in dataset.smiles.values:
            if (smile not in smi_list) and (smile not in smi_for_test.smiles.values):
                smi_list.append(smile)
        random.Random(42).shuffle(smi_list)
        train_smi = smi_list[:int(len(smi_list) * per_train)]
        valid_smi = smi_list[int(len(smi_list) * per_train):]

        train_list, valid_list, test_list = [], [], []
        for item in tqdm(data):
            if item.smi in train_smi:
                train_list.append(item)
            elif item.smi in valid_smi:
                valid_list.append(item)
            else:
                if item.smi in smi_for_test.smiles.values:
                    for index, row in smi_for_test[smi_for_test['smiles'] == item.smi].iterrows():
                        if (row[cond_names].values.astype(np.float32) == item.cond.view(-1).numpy()).all() and (row['y'] == item.y):
                            test_list.append(item)
        logging.info(f'{len(test_list) = }, {len(smi_for_test) = }')
    elif params['SPLITTER'] == 'random':
        per_train = 0.8
        train_val_list, test_list = [], []
        for item in tqdm(data):
            if item.smi in smi_for_test.smiles.values:
                for index, row in smi_for_test[smi_for_test['smiles'] == item.smi].iterrows():
                    if (row[cond_names].values.astype(np.float32) == item.cond.view(-1).numpy()).all() and (row['y'] == item.y):
                        test_list.append(item)
            else:
                train_val_list.append(item)
        random.Random(42).shuffle(train_val_list)
        train_list = train_val_list[:int(len(train_val_list) * per_train)]
        valid_list = train_val_list[int(len(train_val_list) * per_train):]
    else:
        raise Exception('Sorry! Unavailable option')
    loader = DataLoader(train_list,
                        batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
    val_loader = DataLoader(valid_list,
                        batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
    test_loader = DataLoader(test_list,
                        batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
    
    return loader, val_loader, test_loader, train_list, valid_list, test_list

def prepare_loader_two_net(params, dataset, smi_for_test, data, cond_names):
    if params['SPLITTER'] == 'scaffold':
        per_train = 0.8
        smi_list = []
        for smile in dataset.smiles.values:
            if (smile not in smi_list) and (smile not in smi_for_test.smiles.values):
                smi_list.append(smile)
        random.Random(42).shuffle(smi_list)
        train_smi = smi_list[:int(len(smi_list) * per_train)]
        valid_smi = smi_list[int(len(smi_list) * per_train):]

        train_list, valid_list, test_list = [], [], []
        for item in tqdm(data):
            if item[0].smi in train_smi:
                train_list.append(item)
            elif item[0].smi in valid_smi:
                valid_list.append(item)
            else:
                if item[0].smi in smi_for_test.smiles.values:
                    for index, row in smi_for_test[smi_for_test['smiles'] == item[0].smi].iterrows():
                        if (row[cond_names].values.astype(np.float32) == item[0].cond.view(-1).numpy()).all() and (row['y'] == item[0].y):
                            test_list.append(item)
        logging.info(f'{len(test_list) = }, {len(smi_for_test) = }')
    elif params['SPLITTER'] == 'random':
        per_train = 0.8
        train_val_list, test_list = [], []
        for item in tqdm(data):
            if item[0].smi in smi_for_test.smiles.values:
                for index, row in smi_for_test[smi_for_test['smiles'] == item[0].smi].iterrows():
                    if (row[cond_names].values.astype(np.float32) == item[0].cond.view(-1).numpy()).all() and (row['y'] == item[0].y):
                        test_list.append(item)
            else:
                train_val_list.append(item)
        random.Random(42).shuffle(train_val_list)
        train_list = train_val_list[:int(len(train_val_list) * per_train)]
        valid_list = train_val_list[int(len(train_val_list) * per_train):]
    else:
        raise Exception('Sorry! Unavailable option')
    train_dataset = PairDataset(train_list)
    valid_dataset = PairDataset(valid_list)
    test_dataset = PairDataset(test_list)

    loader = DataLoader(train_list, collate_fn=collate,
                        batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
    val_loader = DataLoader(valid_list, collate_fn=collate,
                        batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
    test_loader = DataLoader(test_list, collate_fn=collate,
                        batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)

    return loader, val_loader, test_loader, train_list, valid_list, test_list, train_dataset, valid_dataset, test_dataset

def prepare_training_one_net(device, optimizer, model, loss_fn, loss_r2, loss_mae):
    def train(loader):
        losses = 0
        r2s = 0
        maes = 0
        for i, batch in enumerate(loader):
            # Use GPU
            batch.to(device)
            # Reset gradients
            optimizer.zero_grad()
            # Passing the node features and the connection info
            if model.crafted_add_params_num:
                pred, embedding = model(batch.x.float(), batch.edge_index, batch.edge_weight, batch.batch, batch.cond)
            else:
                pred, embedding = model(batch.x.float(), batch.edge_index, batch.edge_weight, batch.batch)
            loss = torch.sqrt(loss_fn(pred, batch.y))
            losses += loss
            r2s += loss_r2(pred, batch.y)
            maes += loss_mae(pred, batch.y)
            loss.backward()
            # Update using the gradients
            optimizer.step()
        losses_avg = losses / (i+1)
        r2 = r2s / (i+1)
        mae = maes / (i+1)
        return loss, embedding, losses_avg, r2, mae

    def evaluate(val_loader, final_eval_on_original = False):
        losses = 0
        r2s = 0
        maes = 0
        for i, batch in enumerate(val_loader):
            batch.to(device)
            if model.crafted_add_params_num:
                pred, _ = model(batch.x.float(), batch.edge_index, batch.edge_weight, batch.batch, batch.cond)
            else:
                pred, _ = model(batch.x.float(), batch.edge_index, batch.edge_weight, batch.batch)
            loss = torch.sqrt(loss_fn(pred, batch.y))
            losses += loss
            r2s += loss_r2(pred, batch.y)
            maes += loss_mae(pred, batch.y)
        losses_avg = losses / (i+1)
        r2 = r2s / (i+1)
        mae = maes / (i+1)
        return losses_avg, r2, mae
    return train, evaluate

def prepare_training_two_net(device, optimizer, model, loss_fn, loss_r2, loss_mae):
    def train(loader):
        losses = 0
        r2s = 0
        maes = 0
        for i, batch in enumerate(loader):
            # Use GPU
            batch[0].to(device)
            batch[1].to(device)
            # Reset gradients
            optimizer.zero_grad()
            # Passing the node features and the connection info
            if model.crafted_add_params_num:
                pred, embedding = model(batch[0].x.float(), batch[0].edge_index, batch[0].edge_weight, batch[1].x.float(), batch[1].edge_index, batch[1].edge_weight, batch[0].batch, batch[1].batch, batch[0].cond)
            else:
                pred, embedding = model(batch[0].x.float(), batch[0].edge_index, batch[0].edge_weight, batch[1].x.float(), batch[1].edge_index, batch[1].edge_weight, batch[0].batch, batch[1].batch)
            loss = torch.sqrt(loss_fn(pred, batch[0].y))
            losses += loss
            r2s += loss_r2(pred, batch[0].y)
            maes += loss_mae(pred, batch[0].y)
            loss.backward()
            # Update using the gradients
            optimizer.step()
        losses_avg = losses / (i+1)
        r2 = r2s / (i+1)
        mae = maes / (i+1)
        return loss, embedding, losses_avg, r2, mae
     
    def evaluate(val_loader):
        losses = 0
        r2s = 0
        maes = 0
        for i, batch in enumerate(val_loader):
            batch[0].to(device)
            batch[1].to(device)
            if model.crafted_add_params_num:
                pred, _ = model(batch[0].x.float(), batch[0].edge_index, batch[0].edge_weight, batch[1].x.float(), batch[1].edge_index, batch[1].edge_weight, batch[0].batch, batch[1].batch, batch[0].cond)
            else:
                pred, _ = model(batch[0].x.float(), batch[0].edge_index, batch[0].edge_weight, batch[1].x.float(), batch[1].edge_index, batch[1].edge_weight, batch[0].batch, batch[1].batch)
            loss = torch.sqrt(loss_fn(pred, batch[0].y))
            losses += loss
            r2s += loss_r2(pred, batch[0].y)
            maes += loss_mae(pred, batch[0].y)
        losses_avg = losses / (i+1)
        r2 = r2s / (i+1)
        mae = maes / (i+1)
        return losses_avg, r2, mae
    return train, evaluate

def perform_training(train, evaluate, loader, val_loader, scheduler):
    print("Starting training...")
    losses, val_losses, coeffs, val_coeffs, maes, val_maes = [], [], [], [], [], []
    EPOCHS = 300
    for epoch in range(1, EPOCHS + 1):
        loss, h, train_loss, train_r2, train_mae = train(loader)
        losses.append(train_loss.cpu().detach().numpy())
        coeffs.append(train_r2.cpu().detach().numpy())
        maes.append(train_mae.cpu().detach().numpy())
        val_loss, val_r2, val_mae = evaluate(val_loader)
        val_losses.append(val_loss.cpu().detach().numpy())
        val_coeffs.append(val_r2.cpu().detach().numpy())
        val_maes.append(val_mae.cpu().detach().numpy())
        if (epoch) % 10 == 0 or epoch == 1:
            logging.info(f"Epoch {epoch} | Train Loss {train_loss:.3f} | Train MAE {train_mae:.3f} | Train R2 {train_r2:.2f} | Validation Loss {val_loss:.3f} | Validation MAE {val_mae:.3f} | Validation R2 {val_r2:.2f}")
        scheduler.step()

    return losses, val_losses, coeffs, val_coeffs, maes, val_maes

def perform_validation(evaluate, loader, val_loader, test_loader):
    train_final_losses, train_final_r2s, train_final_maes = [], [], []
    val_final_losses, val_final_r2s, val_final_maes = [], [], []
    test_losses, test_r2s, test_maes = [], [], []
    for _ in range(10):
        #train evaluation
        train_final_loss, train_final_r2, train_final_mae = evaluate(loader)
        train_final_losses.append(train_final_loss.cpu().detach().numpy().item())
        train_final_r2s.append(train_final_r2.cpu().detach().numpy().item())
        train_final_maes.append(train_final_mae.cpu().detach().numpy().item())
        #valid evaluation
        val_final_loss, val_final_r2, val_final_mae = evaluate(val_loader)
        val_final_losses.append(val_final_loss.cpu().detach().numpy().item())
        val_final_r2s.append(val_final_r2.cpu().detach().numpy().item())
        val_final_maes.append(val_final_mae.cpu().detach().numpy().item())
        #test evaluation
        test_loss, test_r2, test_mae = evaluate(test_loader)
        test_losses.append(test_loss.cpu().detach().numpy().item())
        test_r2s.append(test_r2.cpu().detach().numpy().item())
        test_maes.append(test_mae.cpu().detach().numpy().item())
    logging.info("\nFinal evaluation")
    logging.info(f"Train loss {statistics.mean(train_final_losses):.3f} +/- {statistics.stdev(train_final_losses):.3f} | Train MAE {statistics.mean(train_final_maes):.3f} +/- {statistics.stdev(train_final_maes):.3f} | Train R2 {statistics.mean(train_final_r2s):.2f} +/- {statistics.stdev(train_final_r2s):.2f}")
    logging.info(f"Valid loss {statistics.mean(val_final_losses):.3f} +/- {statistics.stdev(val_final_losses):.3f} | Valid MAE {statistics.mean(val_final_maes):.3f} +/- {statistics.stdev(val_final_maes):.3f} | Valid R2 {statistics.mean(val_final_r2s):.2f} +/- {statistics.stdev(val_final_r2s):.2f}")
    logging.info(f"Test  loss {statistics.mean(test_losses):.3f} +/- {statistics.stdev(test_losses):.3f} | Test  MAE {statistics.mean(test_maes):.3f} +/- {statistics.stdev(test_maes):.3f} | Test  R2 {statistics.mean(test_r2s):.2f} +/- {statistics.stdev(test_r2s):.2f}")

    return test_losses, test_r2s, test_maes

def plot_losses(losses, val_losses, coeffs, val_coeffs):
    losses_graph = np.array(losses)[:]
    val_losses_graph = np.array(val_losses)[:]

    figl = go.Figure(data=go.Scatter(y=losses_graph))
    figl.add_scatter(y=val_losses_graph)
    figl.show()

    coeffs_graph = np.array(coeffs)
    val_coeffs_graph = np.array(val_coeffs)

    figr = go.Figure(data=go.Scatter(y=coeffs_graph[coeffs_graph > 0]))
    figr.add_scatter(y=val_coeffs_graph[coeffs_graph > 0])
    figr.show()

    logging.info("Training complete")
    
    return None

def final_eval(params, model, device, test_loader, y_dataset_max, y_dataset_min, test_on_original_data = True):
    """
    Evaluate the model on the test dataset and return the original test and predicted values.
    
    Parameters:
    - params: A dictionary containing various parameters for the system.
    - model: The trained model to be evaluated.
    - device: The device on which the model is running.
    - test_loader: The data loader for the test dataset.
    - y_dataset_max: The maximum value in the target variable of the dataset (so normalization can be reversed).
    - y_dataset_min: The minimum value in the target variable of the dataset (so normalization can be reversed).
        test_on_original_data (bool, optional): Whether to evaluate the model on the original data (only important if log-scale was used). Defaults to True.
    
    Returns:
    - original_y_test_np: A numpy array containing the original test values.
    - original_y_pred_np: A numpy array containing the predicted values.
    """
    original_y_test = []
    original_y_pred = []
    for i, batch in enumerate(test_loader):
        if params['ARCHITECTURE'] == 'one-net':
            batch.to(device)
            if model.crafted_add_params_num:
                pred, _ = model(batch.x.float(), batch.edge_index, batch.edge_weight, batch.batch, batch.cond)
            else:
                pred, _ = model(batch.x.float(), batch.edge_index, batch.edge_weight, batch.batch)
        if params['ARCHITECTURE'] == 'two-net':
            batch[0].to(device)
            batch[1].to(device)
            if model.crafted_add_params_num:
                pred, _ = model(batch[0].x.float(), batch[0].edge_index, batch[0].edge_weight, batch[1].x.float(), batch[1].edge_index, batch[1].edge_weight, batch[0].batch, batch[1].batch, batch[0].cond)
            else:
                pred, _ = model(batch[0].x.float(), batch[0].edge_index, batch[0].edge_weight, batch[1].x.float(), batch[1].edge_index, batch[1].edge_weight, batch[0].batch, batch[1].batch)

        if test_on_original_data == True:
            if params['FEATURE_TRANSFORM'] == True:
                ypred_for_eval = torch.exp(pred * (y_dataset_max - y_dataset_min) + y_dataset_min)
                if params['ARCHITECTURE'] == 'one-net': ytrue_for_eval = torch.exp(batch.y * (y_dataset_max - y_dataset_min) + y_dataset_min)
                if params['ARCHITECTURE'] == 'two-net': ytrue_for_eval = torch.exp(batch[0].y * (y_dataset_max - y_dataset_min) + y_dataset_min)
            else:
                ypred_for_eval = pred * (y_dataset_max - y_dataset_min) + y_dataset_min
                if params['ARCHITECTURE'] == 'one-net': ytrue_for_eval = batch.y * (y_dataset_max - y_dataset_min) + y_dataset_min
                if params['ARCHITECTURE'] == 'two-net': ytrue_for_eval = batch[0].y * (y_dataset_max - y_dataset_min) + y_dataset_min
        else:
            ypred_for_eval = pred
            if params['ARCHITECTURE'] == 'one-net': ytrue_for_eval = batch.y
            if params['ARCHITECTURE'] == 'two-net': ytrue_for_eval = batch[0].y

        original_y_pred.extend(ypred_for_eval.cpu().detach().numpy().tolist())
        original_y_test.extend(ytrue_for_eval.float().cpu().detach().numpy().tolist())
    original_y_test_np = np.array([item for sublist in original_y_test for item in sublist])
    original_y_pred_np = np.array([item for sublist in original_y_pred for item in sublist])
    return original_y_test_np, original_y_pred_np

def print_results_of_final_eval(params, model, device, loader, val_loader, test_loader, test_on_original_data = True):
    """
    Evaluates the performance of a  model on a original values from dataset (without normalization) and prints the results.
    
    Args:
        params (dict): A dictionary containing the parameters of the system.
        model (object): The model to be evaluated.
        device (str): The device on which the model is trained (e.g., cpu, cuda).
        loader (object): The data loader for the training dataset.
        val_loader (object): The data loader for the validation dataset.
        test_loader (object): The data loader for the test dataset.
        test_on_original_data (bool, optional): Whether to evaluate the model on the original data (only important if log-scale was used). Defaults to True.
    
    Returns:
        None
    """
    train_orginal_losses, train_original_r2s, train_original_maes, train_original_mares, train_original_ases = [], [], [], [], []
    val_original_losses, val_original_r2s, val_original_maes, val_original_mares, val_original_ases = [], [], [], [], []
    test_original_losses, test_original_r2s, test_original_maes, test_original_mares, test_original_ases = [], [], [], [], []
    evaluator = RegressionMetric()

    for _ in range(10):
        # train evaluation
        original_y_label_train, original_y_pred_train = final_eval(params, model, device, loader, test_on_original_data=test_on_original_data)
        train_orginal_losses.append(evaluator.root_mean_squared_error(original_y_label_train, original_y_pred_train))
        train_original_r2s.append(evaluator.coefficient_of_determination(original_y_label_train, original_y_pred_train))
        train_original_maes.append(evaluator.mean_absolute_error(original_y_label_train, original_y_pred_train))
        train_original_mares.append(evaluator.mean_absolute_percentage_error(original_y_label_train, original_y_pred_train))
        train_original_ases.append(evaluator.a20_index(original_y_label_train, original_y_pred_train))

        # validation evaluation
        original_y_label_valid, original_y_pred_valid = final_eval(params, model, device, val_loader, test_on_original_data=test_on_original_data)
        val_original_losses.append(evaluator.root_mean_squared_error(original_y_label_valid, original_y_pred_valid))
        val_original_r2s.append(evaluator.coefficient_of_determination(original_y_label_valid, original_y_pred_valid))
        val_original_maes.append(evaluator.mean_absolute_error(original_y_label_valid, original_y_pred_valid))
        val_original_mares.append(evaluator.mean_absolute_percentage_error(original_y_label_valid, original_y_pred_valid))
        val_original_ases.append(evaluator.a20_index(original_y_label_valid, original_y_pred_valid))

        # test evaluation
        original_y_label_test, original_y_pred_test = final_eval(params, model, device, test_loader, test_on_original_data=test_on_original_data)
        test_original_losses.append(evaluator.root_mean_squared_error(original_y_label_test, original_y_pred_test))
        test_original_r2s.append(evaluator.coefficient_of_determination(original_y_label_test, original_y_pred_test))
        test_original_maes.append(evaluator.mean_absolute_error(original_y_label_test, original_y_pred_test))
        test_original_mares.append(evaluator.mean_absolute_percentage_error(original_y_label_test, original_y_pred_test))
        test_original_ases.append(evaluator.a20_index(original_y_label_test, original_y_pred_test))

    print(f"\nFinal evaluation {test_on_original_data = }")
    print(f"Train loss {statistics.mean(train_orginal_losses):.3f} +/- {statistics.stdev(train_orginal_losses):.3f} | Train MAE {statistics.mean(train_original_maes):.3f} +/- {statistics.stdev(train_original_maes):.3f} | Train R2 {statistics.mean(train_original_r2s):.2f} +/- {statistics.stdev(train_original_r2s):.2f} | Train MARE {statistics.mean(train_original_mares):.3f} +/- {statistics.stdev(train_original_mares):.3f} | Train A20 {statistics.mean(train_original_ases):.2f} +/- {statistics.stdev(train_original_ases):.2f}")
    print(f"Valid loss {statistics.mean(val_original_losses):.3f} +/- {statistics.stdev(val_original_losses):.3f} | Valid MAE {statistics.mean(val_original_maes):.3f} +/- {statistics.stdev(val_original_maes):.3f} | Valid R2 {statistics.mean(val_original_r2s):.2f} +/- {statistics.stdev(val_original_r2s):.2f} | Valid MARE {statistics.mean(val_original_mares):.3f} +/- {statistics.stdev(val_original_mares):.3f} | Valid A20 {statistics.mean(val_original_ases):.2f} +/- {statistics.stdev(val_original_ases):.2f}")
    print(f"Test  loss {statistics.mean(test_original_losses):.3f} +/- {statistics.stdev(test_original_losses):.3f} | Test  MAE {statistics.mean(test_original_maes):.3f} +/- {statistics.stdev(test_original_maes):.3f} | Test  R2 {statistics.mean(test_original_r2s):.2f} +/- {statistics.stdev(test_original_r2s):.2f} | Test  MARE {statistics.mean(test_original_mares):.3f} +/- {statistics.stdev(test_original_mares):.3f} | Test  A20 {statistics.mean(test_original_ases):.2f} +/- {statistics.stdev(test_original_ases):.2f}")

    figt = go.Figure(data=go.Scatter(x=original_y_label_test, y=original_y_pred_test, mode='markers'))
    figt.update_layout(autosize=False, width=800, height=800,)
    figt.add_shape(type="line", x0=min(original_y_label_test), y0=min(original_y_label_test), x1=max(original_y_label_test), y1=max(original_y_label_test))
    figt.show()