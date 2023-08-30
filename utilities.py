import pandas as pd

def load_dataset(filename='/data/2021_Table04_Datacube.csv', encoding_type='latin-1', index_col=None):
    df = pd.read_csv(filename, encoding=encoding_type, index_col=index_col)
    return df

def load_features_list():
    baseline_cols = [
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
        "Training_MVT_Deposit",                             # Target variable MVT_Deposit
        "Training_MVT_Occurrence",                          # Target variable MVT_Occurrence
    ]
    return baseline_cols


def neighbor_deposits(df):
    # merging Deposit and Occurrence
    df['MVT_Deposit'] = df.apply(lambda row: True if True in [row['Training_MVT_Deposit'], row['Training_MVT_Occurrence']] else False, axis=1)

    #  converting H3_Geometry POLYGON(()) to list of 6 coordinates [(* *), (* *), (* *), (* *), (* *), (* *)]
    df['H3_Geometry2'] = df['H3_Geometry'].apply(lambda x: x[10:-2].split(', ')[:-1])

    # filtering df with MVT_Deposit present
    df_present = df[df['MVT_Deposit']==True] # for MVT there are 2027 rows

    # record all vertices of MVT_Deposit Present polygons
    present_coordinates = [] # -> for MVT 9915 vertices
    for coordinates in df_present['H3_Geometry2']:
        for coordinate in coordinates:
            if coordinate not in present_coordinates:
                present_coordinates.append(coordinate)
    present_coordinates = set(present_coordinates) # converting to set()
                
    # checking if any of 6 vertices of polygon are in present_coordinates
    # if YES then it's a neighbor or itself polygon
    df['MVT_Deposit_wNeighbors'] = df.apply(lambda x: True if (present_coordinates & set(x['H3_Geometry2'])) else False, axis=1)
    df = df.drop(columns=['H3_Geometry2'])
    return df