import pandas as pd
import numpy as np
from math import ceil
from tqdm import tqdm
import rasterio
import rasterio.features
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.model_selection import GroupKFold


def load_dataset(filename='data/2021_Table04_Datacube.csv', encoding_type='latin-1', index_col=None):
    df = pd.read_csv(filename, encoding=encoding_type, index_col=index_col)
    return df


def load_features_dict(deptype='MVT', baseline='baseline'):
    assert deptype in ['MVT','CD']
    assert baseline in ['baseline', 'updated', 'preferred']
    # Note that the order of the created list is very important, particularly for WOE
    if baseline == 'baseline':
        cols = {
            "H3_Geometry": None,                                        # Polygon with coordinates of the vertices
            "Continent_Majority": None,                                 # used to separate US/Canada from Australia
            "Seismic_LAB_Priestley": None,                              # Depth to LAB
            "Seismic_Moho": None,                                       # Depth to Moho
            "Gravity_GOCE_ShapeIndex": None,                            # Sattelite Gravity
            "Gravity_Bouguer": None,                                    # Gravity Bouger
            "Gravity_Bouguer_HGM": None,                                # Gravity HGM
            "Gravity_Bouguer_UpCont30km_HGM": None,                     # Gravity upward cont'd HGM
            "Gravity_Bouguer_HGM_Worms_Proximity": None,                # Gravity worms
            "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": None,     # Gravity upward cont'd worms
            "Magnetic_HGM": None,                                       # Magnetic HGM
            "Magnetic_LongWavelength_HGM": None,                        # Magnetic long-wavelength HGM
            "Magnetic_HGM_Worms_Proximity": None,                       # Magnetic worms
            "Magnetic_LongWavelength_HGM_Worms_Proximity": None,        # Magnetic long-wavelength worms
        }
    elif baseline == 'updated':
        cols = {
            "H3_Geometry": None,                                        # Polygon with coordinates of the vertices
            "Continent_Majority": None,                                 # used to separate US/Canada from Australia
            "Geology_Lithology_Majority": None,                         # Lithology (major) - these seem to be grouped into ~9 categories based on paper
            "Geology_Period_Maximum_Majority": None,                    # Period (maximum) - option 1
            "Geology_Period_Minimum_Majority": None,                    # Period (minimum) - option 1
            # "Geology_Period_Maximum_Minority": None,                  # Period (maximum) - option 2
            # "Geology_Period_Minimum_Minority": None,                  # Period (minimum) - option 2
            "Sedimentary_Dictionary": [                                            # Sedimentary dictionaries
                "Geology_Dictionary_Calcareous",
                "Geology_Dictionary_Carbonaceous",
                "Geology_Dictionary_FineClastic"
            ],  
            "Igneous_Dictionary": [                                                # Igneous dictionaries
                "Geology_Dictionary_Felsic",
                "Geology_Dictionary_Intermediate",
                "Geology_Dictionary_UltramaficMafic"
            ],      
            "Metamorphic_Dictionary": [                                            # Metamorphic dictionaries
                "Geology_Dictionary_Anatectic",
                "Geology_Dictionary_Gneissose",
                "Geology_Dictionary_Schistose"
            ],                 
            "Geology_Paleolatitude_Period_Maximum": None,               # Paleo-latitude
            "Terrane_Proximity": None,                                  # Proximity to terrane boundaries
            "Geology_PassiveMargin_Proximity": None,                    # Proximity to passive margins
            "Geology_BlackShale_Proximity": None,                       # Proximity to black shales
            "Geology_Fault_Proximity": None,                            # Proximity to faults

            "Seismic_LAB_Priestley": None,                              # Depth to LAB                              ??? Why Priestley?
            "Seismic_Moho": None,                                       # Depth to Moho
            "Gravity_GOCE_ShapeIndex": None,                            # Satellite gravity
            "Gravity_Bouguer": None,                                    # Gravity Bouguer
            "Gravity_Bouguer_HGM": None,                                # Gravity HGM
            "Gravity_Bouguer_UpCont30km_HGM": None,                     # Gravity upward continued HGM
            "Gravity_Bouguer_HGM_Worms_Proximity": None,                # Gravity worms
            "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": None,     # Gravity upward continued worms
            "Magnetic_HGM": None,                                       # Magnetic HGM
            "Magnetic_LongWavelength_HGM": None,                        # Magnetic long-wavelength HGM
            "Magnetic_HGM_Worms_Proximity": None,                       # Magnetic worms
            "Magnetic_LongWavelength_HGM_Worms_Proximity": None,        # Magnetic long-wavelength worms
        }
    elif  baseline == 'preferred':
        cols = {
            "H3_Geometry": None,                                        # Polygon with coordinates of the vertices
            "Continent_Majority": None,                                 # used to separate US/Canada from Australia
            "Latitude_EPSG4326": None,                                  # used to split data
            "Geology_Lithology_Majority": None,                         # Lithology (majority)
            "Geology_Lithology_Minority": None,                         # Lithology (minority)
            "Geology_Period_Maximum_Majority": None,                    # Period (maximum) - option 1
            "Geology_Period_Minimum_Majority": None,                    # Period (minimum) - option 1
            # "Geology_Period_Maximum_Minority": None,                  # Period (maximum) - option 2
            # "Geology_Period_Minimum_Minority": None,                  # Period (minimum) - option 2
            "Sedimentary_Dictionary": [                                            # Sedimentary dictionaries
                "Geology_Dictionary_Calcareous",
                "Geology_Dictionary_Carbonaceous",
                "Geology_Dictionary_FineClastic"
            ],  
            "Igneous_Dictionary": [                                                # Igneous dictionaries
                "Geology_Dictionary_Felsic",
                "Geology_Dictionary_Intermediate",
                "Geology_Dictionary_UltramaficMafic"
            ],      
            "Metamorphic_Dictionary": [                                            # Metamorphic dictionaries
                "Geology_Dictionary_Anatectic",
                "Geology_Dictionary_Gneissose",
                "Geology_Dictionary_Schistose"
            ],                 
            "Seismic_LAB_Priestley": None,                              # Depth to LAB                              ??? Why Priestley?
            "Seismic_Moho": None,                                       # Depth to Moho
            "Gravity_GOCE_ShapeIndex": None,                            # Satellite gravity
            "Geology_Paleolatitude_Period_Minimum": None,               # Paleo-latitude                            ??? could be Geology_Paleolatitude_Period_Maximum
            "Terrane_Proximity": None,                                  # Proximity to terrane boundaries
            "Geology_PassiveMargin_Proximity": None,                    # Proximity to passive margins
            "Geology_BlackShale_Proximity": None,                       # Proximity to black shales
            "Geology_Fault_Proximity": None,                            # Proximity to faults
            "Gravity_Bouguer": None,                                    # Gravity Bouguer
            "Gravity_Bouguer_HGM": None,                                # Gravity HGM
            "Gravity_Bouguer_UpCont30km_HGM": None,                     # Gravity upward continued HGM
            "Gravity_Bouguer_HGM_Worms_Proximity": None,                # Gravity worms
            "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": None,     # Gravity upward continued worms
            "Magnetic_HGM": None,                                       # Magnetic HGM
            "Magnetic_LongWavelength_HGM": None,                        # Magnetic long-wavelength HGM
            "Magnetic_HGM_Worms_Proximity": None,                       # Magnetic worms
            "Magnetic_LongWavelength_HGM_Worms_Proximity": None,        # Magnetic long-wavelength worms
        }
    else:
        raise ValueError('Baseline should one of the following: baseline, updated, or preferred')
    
    if deptype == 'MVT':
        cols["Training_MVT_Deposit"] = None                     # Target variable MVT_Deposit
        cols["Training_MVT_Occurrence"] = None                  # Target variable MVT_Occurrence
    elif deptype == 'CD':
        cols["Training_CD_Deposit"] = None                      # Target variable CD_Deposit
        cols["Training_CD_Occurrence"] = None                   # Target variable CD_Occurrence
    return cols


DEFAULT_LITHOLOGY_MAP = {
    "Igneous_Extrusive": "Igneous_Extrusive",
    "Igneous_Intrusive": "Igneous_Intrusive_Felsic",
    "Igneous_Intrusive_Alkalic": "Igneous_Intrusive_Felsic",
    "Igneous_Intrusive_Anorthosite": "Igneous_Intrusive_Felsic",
    "Igneous_Intrusive_Felsic": "Igneous_Intrusive_Felsic",
    "Igneous_Intrusive_Felsic_Felsite": "Igneous_Intrusive_Felsic",
    "Igneous_Intrusive_Felsic_Pegmatite": "Igneous_Intrusive_Felsic",
    "Igneous_Intrusive_Felsic_Tonalite": "Igneous_Intrusive_Felsic",
    "Igneous_Intrusive_Intermediate": "Igneous_Intrusive_Felsic",
    "Igneous_Intrusive_Mafic": "Igneous_Intrusive_Mafic",
    "Igneous_Intrusive_Ultramafic": "Igneous_Intrusive_Mafic",
    "Metamorphic_Amphibolite": "Igneous_Intrusive_Mafic",
    "Metamorphic_Charnockite": "Metamorphic_Gneiss",
    "Metamorphic_Eclogite": "Metamorphic_Gneiss",
    "Metamorphic_Gneiss": "Metamorphic_Gneiss",
    "Metamorphic_Gneiss_Orthogneiss": "Metamorphic_Gneiss", # Metamorphic_Gneiss_Magmatic?
    "Metamorphic_Gneiss_Paragneiss": "Metamorphic_Gneiss_Paragneiss", # Metamorphic_Gneiss_Supracrustal?
    "Metamorphic_Granulite": "Metamorphic_Gneiss",
    "Metamorphic_Marble": "Sedimentary_Chemical",
    "Metamorphic_Migmatite": "Metamorphic_Gneiss",
    "Metamorphic_Quartzite": "Metamorphic_Gneiss_Paragneiss",
    "Metamorphic_Schist": "Metamorphic_Schist",
    "Other_Fault": "Other_Unconsolidated",
    "Other_Hydrothermal": "Other_Unconsolidated",
    "Other_Melange": "Metamorphic_Schist",
    "Other_Unconsolidated": "Other_Unconsolidated",
    "Other_Unknown": "Other_Unconsolidated",
    "Sedimentary_Chemical": "Sedimentary_Chemical",
    "Sedimentary_Chemical_Carbonate": "Sedimentary_Chemical",
    "Sedimentary_Chemical_Evaporite": "Sedimentary_Chemical",
    "Sedimentary_Siliciclastic": "Sedimentary_Siliciclastic",
}


DEFAULT_GEOLOGY_PERIOD_MAP = {
    "Cambrian": "Cambrian",
    "Cretaceous": "Cretaceous",
    "Devonian": "Devonian",
    "Eoarchean": "Neoarchean",
    "Jurassic": "Jurassic",
    "Mesoarchean": "Neoarchean",
    "Mesoproterozoic": "Mesoproterozoic",
    "Mississippian": "Mississippian",
    "Neoarchean": "Neoarchean",
    "Neogene": "Neogene",
    "Neoproterozoic": "Neoproterozoic",
    "Ordovician": "Ordovician",
    "Paleoarchean": "Neoarchean",
    "Paleogene": "Paleogene",
    "Paleoproterozoic": "Paleoproterozoic",
    "Pennsylvanian": "Pennsylvanian",
    "Permian": "Permian",
    "Quaternary": "Quaternary",
    "Silurian": "Silurian",
    "Triassic": "Triassic",
}


def extract_cols(df, cols_dict, lith_map=DEFAULT_LITHOLOGY_MAP, perd_map=DEFAULT_GEOLOGY_PERIOD_MAP):
    # extracts the feature columns from the datacube, re-binning where necessary
    # NOTE - we may need to remap lithology minority ONLY
    out_df = pd.DataFrame()
    for col_name, col_items in cols_dict.items():
        if "LITHOLOGY" in col_name.upper():
            # special "re-binning" should be applied to lithology data
            out_df[col_name] = df[col_name].replace(lith_map)
        elif "GEOLOGY_PERIOD" in col_name.upper():
            # special "re-binning" should be applied to geology period data
            out_df[col_name] = df[col_name].replace(perd_map)
        else:    
            if col_items is None:
                # existing column, no merging required
                out_df[col_name] = df[col_name]
            else:
                # special "re-binning" applies to geology columns - sedimentary, igneous, metamorphic
                out_df[col_name] = df.loc[:,col_items].any(axis=1)
    return out_df, list(cols_dict.keys())


def neighbor_deposits(df, deptype='MVT'):
    assert deptype in ['MVT','CD']
    # merging Deposit and Occurrence
    df[f'{deptype}_Deposit'] = df.apply(lambda row: True if True in [row[f'Training_{deptype}_Deposit'], row[f'Training_{deptype}_Occurrence']] else False, axis=1)

    #  converting H3_Geometry POLYGON(()) to list of 6 coordinates [(* *), (* *), (* *), (* *), (* *), (* *)]
    df['H3_Geometry2'] = df['H3_Geometry'].apply(lambda x: x[10:-2].split(', ')[:-1])

    # filtering df with MVT_Deposit present
    df_present = df[df[f'{deptype}_Deposit']==True] # for MVT there are 2027 rows

    # record all vertices of MVT_Deposit Present polygons
    present_coordinates = [] # -> for MVT 9915 vertices
    for coordinates in df_present['H3_Geometry2']:
        for coordinate in coordinates:
            if coordinate not in present_coordinates:
                present_coordinates.append(coordinate)
    present_coordinates = set(present_coordinates) # converting to set()
                
    # checking if any of 6 vertices of polygon are in present_coordinates
    # if YES then it's a neighbor or itself polygon
    df[f'{deptype}_Deposit_wNeighbors'] = df.apply(lambda x: True if (present_coordinates & set(x['H3_Geometry2'])) else False, axis=1)
    df = df.drop(columns=['H3_Geometry2'])
    return df


def tukey_remove_outliers(df, multiplier=1.5, replacement_percentile=0.05):
    for col in df.columns:
        if df[col].dtype != "float64": continue
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
        if df[col].isna().sum():
            if df[col].dtype != "float64" and "Geology_Period" in col:
                df[col].fillna("Quaternary", inplace=True)
            elif df[col].dtype != "float64":
                df[col].fillna("UNK", inplace=True)
            else:
                df[col].fillna(value=df[col].mean(), inplace=True)
    return df


def normalize_df(df):
    # standardizes the data
    for col in df.columns:
        if df[col].dtype != "float64": continue
        df[col] = (df[col]-df[col].mean()) / df[col].std()
    return df


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


def get_spatial_cross_val_idx(df, k=5, test_set=0, split_col="target", nbins=None, samples_per_bin=3.0):
    # select only the deposit/occurence/neighbor present samples
    target_df = df.loc[df[split_col] == True,"Latitude_EPSG4326"]
    # bin the latitudes into sizes of 1-3 samples per bin
    if nbins is None:
        nbins = ceil(len(target_df) / samples_per_bin)
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
    test_df = df[df["group"] == test_set]
    train_df = df[df["group"] != test_set]
    # generate a group k-fold sampling
    group_kfold = GroupKFold(n_splits=k)
    train_idx = group_kfold.split(train_df, train_df["target"], train_df["group"])
    return test_df, train_df, train_idx


def convert_categorical(df, category_col):
    categories = np.unique(df[category_col]).tolist()
    categories_dict = {category: float(idx) for idx, category in enumerate(categories)}
    df[category_col] = df[category_col].replace(categories_dict).astype("uint8")
    return df
    

def rasterize_datacube(datacube, meta, data_dir, region):
    tif_layers = [col for col in datacube.columns.to_list() if ("Continent" not in col) and ("H3" not in col)]
    meta.update(tiled=True)
    meta.update(blockxsize=32)
    meta.update(blockysize=32)
    print(f"Outputting - {tif_layers}")

    for idx, tif_layer in tqdm(enumerate(tif_layers), total=len(tif_layers)):
        datacube_tif_file = f"{data_dir}datacube_{region}_{tif_layer.lower().replace(' ','-')}.tif"
        with rasterio.open(datacube_tif_file, "w", **meta) as out:
            # converts categoricals to ints
            if datacube[tif_layer].dtype != "float64" and datacube[tif_layer].dtype != "bool":
                datacube = convert_categorical(datacube, tif_layer)
            
            # this is where we create a generator of geom, value pairs to use in rasterizing
            shapes = list(datacube.loc[:,["H3_Geometry", tif_layer]].itertuples(index=False, name=None))
            burned = rasterio.features.rasterize(
                shapes=shapes,
                out_shape=(meta["height"],meta["width"]),
                fill=meta["nodata"],
                transform=out.transform,
            )

            # writes the n-dim tif
            out.write_band(1, burned)


def visualize_datacube(datacube, meta):
    tif_layers = [col for col in datacube.columns.to_list() if ("Continent" not in col) and ("H3" not in col)]
    print(f"Plotting - {tif_layers}")

    for idx, tif_layer in tqdm(enumerate(tif_layers), total=len(tif_layers)):
        # converts categoricals to ints
        if datacube[tif_layer].dtype != "float64" and datacube[tif_layer].dtype != "bool":
            datacube = convert_categorical(datacube, tif_layer)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = list(datacube.loc[:,["H3_Geometry",tif_layer]].itertuples(index=False, name=None))
        burned = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(meta["height"],meta["width"]),
            fill=np.nan,
            transform=meta["transform"]
        )

        if datacube[tif_layer].dtype == "bool":
            cmap = mpl.colors.ListedColormap(['black', 'red'])
            plt.imshow(burned, cmap=cmap)
        else:
            plt.imshow(burned, cmap="turbo")
            plt.colorbar()
        
        plt.title(tif_layer)
        plt.show()
