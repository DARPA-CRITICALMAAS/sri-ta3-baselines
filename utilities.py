import pandas as pd
import numpy as np
from math import ceil

from sklearn.model_selection import GroupKFold


def load_dataset(filename='data/2021_Table04_Datacube.csv', encoding_type='latin-1', index_col=None):
    df = pd.read_csv(filename, encoding=encoding_type, index_col=index_col)
    return df


def load_features_list(type=['MVT','CD'], baseline=['baseline', 'updated', 'preferred']):
    # Note that the order of the created list is very important, particularly for WOE
    if baseline == 'baseline':
        cols = [
            "H3_Geometry",                                      # Polygon with coordinates of the vertices
            "Seismic_LAB_Priestley",                            # Depth to LAB
            "Seismic_Moho",                                     # Depth to Moho
            "Gravity_GOCE_ShapeIndex",                          # Sattelite Gravity
            "Gravity_Bouguer",                                  # Gravity Bouger
            "Gravity_Bouguer_HGM",                              # Gravity HGM
            "Gravity_Bouguer_UpCont30km_HGM",                   # Gravity upward cont'd HGM
            "Gravity_Bouguer_HGM_Worms_Proximity",              # Gravity worms
            "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity",   # Gravity upward cont'd worms
            "Magnetic_HGM",                                     # Magnetic HGM
            "Magnetic_LongWavelength_HGM",                      # Magnetic long-wavelength HGM
            "Magnetic_HGM_Worms_Proximity",                     # Magnetic worms
            "Magnetic_LongWavelength_HGM_Worms_Proximity",      # Magnetic long-wavelength worms
        ]
    elif baseline == 'updated':
        cols = [
            "H3_Geometry",                                      # Polygon with coordinates of the vertices
            "Geology_Lithology_Majority",                       # Lithology (major)
            "",                 # Period (maximum)
            "",                 # Period (minimum)
            "Geology_Dictionary_Sedimentary",                   # Sedimentary dictionaries
            "Geology_Dictionary_Igneous",                       # Igneous dictionaries
            "",                 # Metamorphic dictionaries
            "Seismic_LAB_Priestley",                            # Depth to LAB                              ??? Why Priestley?
            "Seismic_Moho",                                     # Depth to Moho
            "Gravity_GOCE_ShapeIndex",                          # Satellite gravity
            "Geology_Paleolatitude_Period_Minimum",             # Paleo-latitude                            ??? could be Geology_Paleolatitude_Period_Maximum
            "Terrane_Proximity",                                # Proximity to terrane boundaries
            "Geology_PassiveMargin_Proximity",                  # Proximity to passive margins
            "Geology_BlackShale_Proximity",                     # Proximity to black shales
            "Geology_Fault_Proximity",                          # Proximity to faults
            "Gravity_Bouguer",                                  # Gravity Bouguer
            "Gravity_Bouguer_HGM",                              # Gravity HGM
            "Gravity_Bouguer_UpCont30km_HGM",                   # Gravity upward continued HGM
            "Gravity_Bouguer_HGM_Worms_Proximity",              # Gravity worms
            "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity",   # Gravity upward continued worms
            "Magnetic_HGM",                                     # Magnetic HGM
            "Magnetic_LongWavelength_HGM",                      # Magnetic long-wavelength HGM
            "Magnetic_HGM_Worms_Proximity",                     # Magnetic worms
            "Magnetic_LongWavelength_HGM_Worms_Proximity",      # Magnetic long-wavelength worms
        ]
    elif  baseline == 'preferred':
        cols = [
            "H3_Geometry",                                      # Polygon with coordinates of the vertices
            "Geology_Lithology_Majority",                       # Lithology (major)
            "Geology_Lithology_Minority",                       # Lithology (minority)
            "",                 # Period (maximum)
            "",                 # Period (minimum)
            "Geology_Dictionary_Sedimentary",                   # Sedimentary dictionaries
            "Geology_Dictionary_Igneous",                       # Igneous dictionaries
            "",                 # Metamorphic dictionaries
            "Seismic_LAB_Priestley",                            # Depth to LAB                              ??? Why Priestley?
            "Seismic_Moho",                                     # Depth to Moho
            "Gravity_GOCE_ShapeIndex",                          # Satellite gravity
            "Geology_Paleolatitude_Period_Minimum",             # Paleo-latitude                            ??? could be Geology_Paleolatitude_Period_Maximum
            "Terrane_Proximity",                                # Proximity to terrane boundaries
            "Geology_PassiveMargin_Proximity",                  # Proximity to passive margins
            "Geology_BlackShale_Proximity",                     # Proximity to black shales
            "Geology_Fault_Proximity",                          # Proximity to faults
            "Gravity_Bouguer",                                  # Gravity Bouguer
            "Gravity_Bouguer_HGM",                              # Gravity HGM
            "Gravity_Bouguer_UpCont30km_HGM",                   # Gravity upward continued HGM
            "Gravity_Bouguer_HGM_Worms_Proximity",              # Gravity worms
            "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity",   # Gravity upward continued worms
            "Magnetic_HGM",                                     # Magnetic HGM
            "Magnetic_LongWavelength_HGM",                      # Magnetic long-wavelength HGM
            "Magnetic_HGM_Worms_Proximity",                     # Magnetic worms
            "Magnetic_LongWavelength_HGM_Worms_Proximity",      # Magnetic long-wavelength worms
        ]
    else:
        raise ValueError('Baseline should one of the following: baseline, updated, or preferred')
    
    if type == 'MVT':
        cols.append("Training_MVT_Deposit")                     # Target variable MVT_Deposit
        cols.append("Training_MVT_Occurrence")                  # Target variable MVT_Occurrence
    elif type == 'CD':
        cols.append("Training_CD_Deposit")                      # Target variable CD_Deposit
        cols.append("Training_CD_Occurrence")                   # Target variable CD_Occurrence
    else:
        raise ValueError('Deposit types are either MVT or CD')
    return cols


def neighbor_deposits(df, type=['MVT','CD']):
    # merging Deposit and Occurrence
    df[f'{type}_Deposit'] = df.apply(lambda row: True if True in [row[f'Training_{type}_Deposit'], row[f'Training_{type}_Occurrence']] else False, axis=1)

    #  converting H3_Geometry POLYGON(()) to list of 6 coordinates [(* *), (* *), (* *), (* *), (* *), (* *)]
    df['H3_Geometry2'] = df['H3_Geometry'].apply(lambda x: x[10:-2].split(', ')[:-1])

    # filtering df with MVT_Deposit present
    df_present = df[df[f'{type}_Deposit']==True] # for MVT there are 2027 rows

    # record all vertices of MVT_Deposit Present polygons
    present_coordinates = [] # -> for MVT 9915 vertices
    for coordinates in df_present['H3_Geometry2']:
        for coordinate in coordinates:
            if coordinate not in present_coordinates:
                present_coordinates.append(coordinate)
    present_coordinates = set(present_coordinates) # converting to set()
                
    # checking if any of 6 vertices of polygon are in present_coordinates
    # if YES then it's a neighbor or itself polygon
    df[f'{type}_Deposit_wNeighbors'] = df.apply(lambda x: True if (present_coordinates & set(x['H3_Geometry2'])) else False, axis=1)
    df = df.drop(columns=['H3_Geometry2'])
    return df


def tukey_remove_outliers(df, multiplier=1.5, replacement_percentile=0.05):
    for col in df.columns:
        # get the IQR
        Q1 = df.loc[:,col].quantile(0.25)
        Q3 = df.loc[:,col].quantile(0.75)
        IQR = Q3 - Q1
        # get the lower bound replacements and replace the values
        P05 = df.loc[:,col].quantile(replacement_percentile)
        mask = df.loc[:,col] < (Q1 - multiplier * IQR)
        df.loc[mask, col] = P05
        # get the upper bound replacements and replace the values
        P95 = df.loc[:,col].quantile(1.0-replacement_percentile)
        mask = df.loc[:,col] > (Q3 + multiplier * IQR)
        df.loc[mask, col] = P95
    return df


def impute_nans(df):
    # fills nan values with mean, for each column
    for col in df.columns:
        df[col].fillna(value=df[col].mean(), inplace=True)
    return df


def normalize_df(df):
    # standardizes the data
    return (df-df.mean()) / df.std()


def calculate_woe_iv(dataset, feature, target):
    # WOE and IV computation from online
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature]
        }) 
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    dset = dset.sort_values(by='WoE')
    return dset, iv


def get_spatial_cross_val_idx(df, k=5):
    # select only the deposit/occurence/neighbor present samples
    target_df = df.loc[df["target"] == True,"Latitude_EPSG4326"]
    # sort the latitudes
    target_df = target_df.sort_values(ignore_index=True)
    # bin the latitudes into sizes of 1-3 samples per bin
    nbins = ceil(len(target_df) / 3.0)
    _, bins = pd.qcut(target_df, nbins, retbins=True)
    bins[0] = -float("inf")
    bins[-1] = float("inf")
    bins = pd.IntervalIndex.from_breaks(bins)
    # group the bins into k+1 groups (folds) - +1 is for test set
    bins_df = pd.DataFrame({"Latitude_EPSG4326": bins})
    bins_df["group"] = np.tile(np.arange(k+1), (ceil(nbins / k+1),))[:nbins]
    # assign all data to a k+1 group using the existing bin / group assignments
    df["Latitude_EPSG4326"] = pd.cut(df["Latitude_EPSG4326"], bins)
    df = pd.merge(df, bins_df, on="Latitude_EPSG4326")
    # split into train / test data
    test_df = df[df["group"] == k]
    train_df = df[df["group"] < k]
    # generate a group k-fold sampling
    group_kfold = GroupKFold(n_splits=k)
    train_idx = group_kfold.split(train_df, train_df["target"], train_df["group"])
    return test_df, train_df, train_idx