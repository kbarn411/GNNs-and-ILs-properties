import pandas as pd

def prepare_density(clean_dataset):
    ions = pd.read_excel('density.xlsx', sheet_name='S2 | Ions')
    ions = ions.iloc[: , [1,3]]
    ions_dict = ions.set_index('Abbreviation').T.to_dict('index')['SMILES']
    densities = pd.read_excel('density.xlsx', sheet_name='S8 | Modeling vs "raw" database')
    densities = densities.iloc[:, [2,3,6,7,8,9,10]]
    if clean_dataset:
        densities.dropna(inplace=True)
        densities.drop(densities[densities['Excluded IL'] != 'no'].index, inplace = True)
        densities.drop(densities[densities['Accepted dataset'] != 'yes'].index, inplace = True)
        densities.reset_index(drop=True, inplace=True)
    densities['cation smiles'] = densities['Cation'].map(ions_dict)
    densities['anion smiles'] = densities['Anion'].map(ions_dict)
    densities['IL'] = '[' + densities['Cation'] + '][' + densities['Anion'] + ']'
    densities['smiles'] = densities['cation smiles'] + '.' + densities['anion smiles']
    dataset = densities.iloc[:, 4:]
    dataset_renamer = {'T / K': 'T_K', 'p / MPa': 'P_MPa', 'ρ / kg/m3': 'd_kg*m-3', 'cation smiles': 'cation', 'anion smiles': 'anion'}
    dataset.rename(columns = dataset_renamer, inplace=True)
    col_order = ['IL', 'cation', 'anion', 'd_kg*m-3', 'T_K', 'P_MPa', 'smiles']
    dataset = dataset[col_order]
    if clean_dataset:
        dataset.to_csv('./data-density-clean.csv', sep=',', index = False)
    else:
        dataset.to_csv('./data-density.csv', sep=',', index = False)

def prepare_viscosity(clean_dataset):
    ions = pd.read_excel('viscosity.xlsx', sheet_name='S2 | Ions')
    ions = ions.iloc[: , [1,3]]
    ions_dict = ions.set_index('Abbreviation').T.to_dict('index')['SMILES']
    viscosities = pd.read_excel('viscosity.xlsx', sheet_name='S8 | Modeling vs "raw" database')
    viscosities = viscosities.iloc[:, [2,3,6,7,8,9]]
    if clean_dataset:
        viscosities.dropna(inplace=True)
        viscosities.drop(viscosities[viscosities['Excluded IL'] != False].index, inplace = True)
        viscosities.drop(viscosities[viscosities['Accepted dataset'] != False].index, inplace = True)
        viscosities.reset_index(drop=True, inplace=True)
    viscosities['cation smiles'] = viscosities['Cation'].map(ions_dict)
    viscosities['anion smiles'] = viscosities['Anion'].map(ions_dict)
    viscosities['IL'] = '[' + viscosities['Cation'].astype('str') + '][' + viscosities['Anion'].astype('str') + ']'
    viscosities['smiles'] = viscosities['cation smiles'].astype('str') + '.' + viscosities['anion smiles'].astype('str')
    dataset = viscosities.iloc[:, 4:]
    dataset_renamer = {'T / K': 'T_K', 'η / mPa s': 'n_mPas', 'cation smiles': 'cation', 'anion smiles': 'anion'}
    dataset.rename(columns = dataset_renamer, inplace=True)
    col_order = ['IL', 'cation', 'anion', 'n_mPas', 'T_K', 'smiles']
    dataset = dataset[col_order]
    if clean_dataset:
        dataset.to_csv('./data-viscosity-clean.csv', sep=',', index = False)
    else:
        dataset.to_csv('./data-viscosity.csv', sep=',', index = False)

def prepare_surface(clean_dataset):
    ions = pd.read_excel('surface_tension.xlsx', sheet_name='S2 | Ions')
    ions = ions.iloc[: , [1,3]]
    ions_dict = ions.set_index('Abbreviation').T.to_dict('index')['SMILES']
    tensions = pd.read_excel('surface_tension.xlsx', sheet_name='S9 | Modeling vs "raw" database')
    tensions = tensions.iloc[:, [2,3,6,7,8,9]]
    if clean_dataset:
        tensions.dropna(inplace=True)
        tensions.drop(tensions[tensions['Excluded IL'] != False].index, inplace = True)
        tensions.drop(tensions[tensions['Accepted dataset'] != False].index, inplace = True)
        tensions.reset_index(drop=True, inplace=True)
    tensions['cation smiles'] = tensions['Cation'].map(ions_dict)
    tensions['anion smiles'] = tensions['Anion'].map(ions_dict)
    tensions['IL'] = '[' + tensions['Cation'].astype('str') + '][' + tensions['Anion'].astype('str') + ']'
    tensions['smiles'] = tensions['cation smiles'].astype('str') + '.' + tensions['anion smiles'].astype('str')
    dataset = tensions.iloc[:, 4:]
    dataset_renamer = {'T / K': 'T_K', 'σ / mN/m': 's_mNm', 'cation smiles': 'cation', 'anion smiles': 'anion'}
    dataset.rename(columns = dataset_renamer, inplace=True)
    col_order = ['IL', 'cation', 'anion', 's_mNm', 'T_K', 'smiles']
    dataset = dataset[col_order]
    if clean_dataset:
        dataset.to_csv('./data-surface-clean.csv', sep=',', index = False)
    else:
        dataset.to_csv('./data-surface.csv', sep=',', index = False)

prepare_density(False)
prepare_density(True)
prepare_viscosity(False)
prepare_viscosity(True)
prepare_surface(False)
prepare_surface(True)